def ask_yes_no(question):
    while True:
        choice = input("{} [y/N]: ".format(question)).lower()
        if choice in ["y", "ye", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False
