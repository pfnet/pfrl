import binascii
import collections
import os
import pickle
from datetime import datetime
from struct import calcsize, pack, unpack

from pfrl.collections.random_access_queue import RandomAccessQueue

# code for future extension. `_VanillaFS` is a dummy of chainerio's
# FIleSystem class.
_VanillaFS = collections.namedtuple("_VanillaFS", "exists open makedirs")
_chainerio_available = False


_INDEX_FILENAME_FORMAT = "chunk.{}.idx"
_DATA_FILENAME_FORMAT = "chunk.{}.data"


class _ChunkReader(object):
    def __init__(self, datadir, fs):
        self.datadir = datadir
        self.fs = fs

    def read_chunk_index(self, gen):
        index_format = _ChunkWriter.index_format
        index_format_size = _ChunkWriter.index_format_size
        indexfile = os.path.join(self.datadir, _INDEX_FILENAME_FORMAT.format(gen))
        with self.fs.open(indexfile, "rb") as ifp:
            idata = ifp.read()
        offset = 0
        while True:
            # TODO: try iter_unpack()
            buf = idata[offset : offset + index_format_size]

            if len(buf) != index_format_size:
                break
            data = unpack(index_format, buf)
            offset += index_format_size
            yield data

    def read_chunk(self, gen, do_unpickle=True):
        datafile = os.path.join(self.datadir, _DATA_FILENAME_FORMAT.format(gen))

        indices = self.read_chunk_index(gen)
        with self.fs.open(datafile, "rb") as dfp:
            # TODO: inefficient; chunked read
            cdata = dfp.read()

        for index in indices:
            g, o, l, c, _ = index
            data = cdata[o : o + l]
            crc = binascii.crc32(data)
            assert crc == c
            if do_unpickle:
                data = pickle.loads(data)
            yield data

    def _count_all_chunks(self):
        gen = 0
        while True:
            indexfile = os.path.join(self.datadir, _INDEX_FILENAME_FORMAT.format(gen))
            datafile = os.path.join(self.datadir, _DATA_FILENAME_FORMAT.format(gen))
            if self.fs.exists(indexfile) and self.fs.exists(datafile):
                count = len(list(self.read_chunk_index(gen)))
                yield gen, count
                gen += 1
                continue
            break

    def read_chunks(self, maxlen, buf):
        """Efficiently read all data needed (but scans all index)"""
        gens = []
        chunks = list(self._count_all_chunks())
        # chunks: [(0, 1024), (1, 1024), ..., (n, 1024)]
        chunks.reverse()
        remain = maxlen
        for gen, count in chunks:
            if maxlen is None:
                gens.append(gen)
            elif remain > 0:
                gens.append(gen)
                remain -= count
            else:
                break
        # gens: [n, n-1, ..., m]
        gens.reverse()
        for gen in gens:
            buf.extend(obj for obj in self.read_chunk(gen))

        gen = max(gens) + 1 if gens else 0
        return gen


class _ChunkWriter(object):
    index_format = "QQQIi"
    index_format_size = calcsize(index_format)

    def __init__(self, datadir, gen, chunksize, fs, do_pickle=True):
        self.datadir = datadir
        self.gen = gen
        assert gen >= 0
        # Threshold should be number of elements or chunksize?
        self.chunksize = chunksize
        assert chunksize > 0
        self.do_pickle = do_pickle
        self.fs = fs

        # AppendOnly
        self.indexfile = os.path.join(datadir, _INDEX_FILENAME_FORMAT.format(gen))
        self.ifp = self.fs.open(self.indexfile, "wb")
        self.datafile = os.path.join(datadir, _DATA_FILENAME_FORMAT.format(gen))
        self.dfp = self.fs.open(self.datafile, "wb")

        self.offset = 0
        self.full = self.chunksize < self.offset

    def is_full(self):
        return self.full

    def append(self, data):
        """
        From https://github.com/kuenishi/machi-py/blob/master/machi/machi.py

        Index entry format:
        0       8       16      24  28  32bytes
        +-------+-------+-------+---+---+
        |gen    |offset |length |crc|st |
        +-------+-------+-------+---+---+
        Indices are appended to index files. Number of max entries per index
        file is to be preset.
        """
        if self.is_full():
            raise RuntimeError("Already chunk written full")

        if self.do_pickle:
            data = pickle.dumps(data)

        self.dfp.write(data)
        self.dfp.flush()
        crc = binascii.crc32(data)
        length = len(data)
        index = pack(self.index_format, self.gen, self.offset, length, crc, 0)
        self.ifp.write(index)
        self.ifp.flush()

        self.offset += length
        self.full = self.chunksize < self.offset
        if self.is_full():
            self.close()

    def __del__(self):
        if not self.full:
            self.close()

    def close(self):
        self.full = True
        self.dfp.close()
        self.ifp.close()


class PersistentRandomAccessQueue(object):
    """Persistent data structure for replay buffer

    Features:
    - Perfectly compatible with collections.RandomAccessQueue
    - Persist replay buffer data on storage to survive restart
      - [todo] remove popleft'd data from disk
    - Reuse replay buffer data to another training session
      - Track back the replay buffer lineage
    Non-it-is-for:
    - Extend replay buffer by spilling to the disk

    TODOs
    - Optimize writes; buffered writes with threads or something

    Arguments:
        basedir (str): Path to the directory where replay buffer data is stored.
        maxlen (int): Max length of queue. Appended data beyond
            this limit is only removed from memory, not from storage.
        ancestor (str): Path to pre-generated replay buffer.
        logger: logger

    """

    def __init__(self, basedir, maxlen, *, ancestor=None, logger=None):
        assert maxlen is None or maxlen > 0
        self.basedir = basedir
        self._setup_fs(None)
        self._setup_datadir()
        self.meta = None
        self.buffer = RandomAccessQueue(maxlen=maxlen)
        self.logger = logger
        self.ancestor_meta = None
        if ancestor is not None:
            # Load ancestor as preloaded data
            meta = self._load_ancestor(ancestor, maxlen)
            self.ancestor_meta = meta

        # Load or create meta file and share the meta object
        self.meta_file = PersistentRandomAccessQueue._meta_file_name(self.basedir)
        self._load_meta(ancestor, maxlen)

        if self.fs.exists(self.datadir):
            reader = _ChunkReader(self.datadir, self.fs)
            self.gen = reader.read_chunks(maxlen, self.buffer)

        else:
            self.gen = 0
            self.fs.makedirs(self.datadir, exist_ok=True)

        self.tail = _ChunkWriter(
            self.datadir, self.gen, self.chunk_size, self.fs, do_pickle=True
        )  # Last chunk to be appended
        self.gen += 1

        if self.logger:
            self.logger.info(
                "Initial buffer size=%d, next gen=%d", len(self.buffer), self.gen
            )

    def _load_meta(self, ancestor, maxlen):
        # This must be checked by single process to avoid race
        # condition where one creates and the other may detect it
        # as exisiting... process differently OTL
        if self.fs.exists(self.meta_file):
            # Load existing meta
            with self.fs.open(self.meta_file, "rb") as fp:
                self.meta = pickle.load(fp)

            # TODO: update chunksize and other properties
            assert isinstance(self.meta, dict)

            # MPI world size must be the same when it's restart
            assert (
                self.meta["comm_size"] == self.comm_size
            ), "Reloading same basedir requires same comm.size"

        else:
            # Create meta from scratch
            # Timestamp from pfrl.experiments.prepare_output_dir

            ts = datetime.strftime(datetime.today(), "%Y%m%dT%H%M%S.%f")
            self.meta = dict(
                basedir=self.basedir,
                maxlen=maxlen,
                comm_size=self.comm_size,
                ancestor=ancestor,
                timestamp=ts,
                chunksize=self.chunk_size,
                trim=False,  # `trim` is reserved for future extension.
            )

            # Note: If HDFS access fails at first open, make sure
            # no ``cv2`` import fail happening - failing
            # ``opencv-python`` due to lacking ``libSM.so`` may
            # break whole dynamic library loader and thus breaks
            # other dynamic library loading (e.g. libhdfs.so)
            # which may happen here. Solution for this is to let
            # the import success, e.g. installing the lacking
            # library correctly.
            self.fs.makedirs(self.basedir, exist_ok=True)
            with self.fs.open(self.meta_file, "wb") as fp:
                pickle.dump(self.meta, fp)

    def close(self):
        self.tail.close()
        self.tail = None

    def _append(self, value):
        if self.tail.is_full():
            self.tail = _ChunkWriter(
                self.datadir, self.gen, self.chunk_size, self.fs, do_pickle=True
            )
            if self.logger:
                self.logger.info("Chunk rotated. New gen=%d", self.gen)
            self.gen += 1

        self.tail.append(value)

    # RandomAccessQueue-compat methods
    def append(self, value):
        self._append(value)
        self.buffer.append(value)

    def extend(self, xs):
        for x in xs:
            self._append(x)
        self.buffer.extend(xs)

    def __iter__(self):
        return iter(self.buffer)

    def __repr__(self):
        return "PersistentRandomAccessQueue({})".format(str(self.buffer))

    def __setitem__(self, i, x):
        raise NotImplementedError()

    def __getitem__(self, i):
        return self.buffer[i]

    def sample(self, n):
        return self.buffer.sample(n)

    def popleft(self):
        self.buffer.popleft()

    def __len__(self):
        return len(self.buffer)

    @property
    def maxlen(self):
        return self.meta["maxlen"]

    @property
    def comm_size(self):
        return 1  # Fixed to 1

    @property
    def comm_rank(self):
        return 0

    @property
    def chunk_size(self):
        return 16 * 128 * 1024 * 1024  # Fixed: 16 * 128MB

    @staticmethod
    def _meta_file_name(dirname):
        return os.path.join(dirname, "meta.pkl")

    def _setup_fs(self, fs):
        # In __init__() fs is fixed to None, but this is reserved for
        # future extension support non-posix file systems such as HDFS
        if fs is None:
            if _chainerio_available:
                # _chainerio_available must be None for now
                raise NotImplementedError(
                    "Internal Error: chainerio support is not yet implemented"
                )
            else:
                # When chainerio is not installed
                self.fs = _VanillaFS(
                    open=open, exists=os.path.exists, makedirs=os.makedirs
                )
        else:
            self.fs = fs

    def _setup_datadir(self):
        # the name "rank0" means that the process is the rank 0
        # in a parallel processing process group
        # It is fixed to 'rank0' and prepared for future extension.
        self.datadir = os.path.join(self.basedir, "rank0")

    def _load_ancestor(self, ancestor, num_data_needed):
        """Simple implementation"""
        ancestor_metafile = PersistentRandomAccessQueue._meta_file_name(ancestor)
        with self.fs.open(ancestor_metafile, "rb") as fp:
            meta = pickle.load(fp)
            assert isinstance(meta, dict)
        if self.logger:
            self.logger.info("Loading buffer data from %s", ancestor)

        datadirs = []
        saved_comm_size = meta["comm_size"]
        n_data_dirs = (saved_comm_size + self.comm_size - 1) // self.comm_size
        data_dir_i = self.comm_rank
        for _ in range(n_data_dirs):
            data_dir_i = data_dir_i % saved_comm_size
            datadirs.append(os.path.join(ancestor, "rank{}".format(data_dir_i)))
            data_dir_i += self.comm_size

        length = 0
        for datadir in datadirs:
            reader = _ChunkReader(datadir, self.fs)
            gen = 0
            while True:
                filename = os.path.join(datadir, _INDEX_FILENAME_FORMAT.format(gen))
                if not self.fs.exists(filename):
                    break
                if self.logger:
                    self.logger.debug("read_chunk_index from %s, gen=%d", datadir, gen)
                length += len(list(reader.read_chunk_index(gen)))
                gen += 1

        if length < num_data_needed and meta["ancestor"] is not None:
            self._load_ancestor(meta["ancestor"], num_data_needed - length)

        for datadir in datadirs:
            reader = _ChunkReader(datadir, self.fs)
            rank_data = []
            maxlen = num_data_needed - len(self.buffer)
            if maxlen <= 0:
                break
            _ = reader.read_chunks(maxlen, rank_data)
            if self.logger:
                self.logger.info(
                    "%d data loaded to buffer (rank=%d)", len(rank_data), self.comm_rank
                )
            self.buffer.extend(rank_data)
        return meta
