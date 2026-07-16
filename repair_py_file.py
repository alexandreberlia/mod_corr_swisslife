"""

Utilitaire de récupération des fichiers Python corrompus.

Cas d'usage :
-------------
Certains exports Jupyter ou transferts de fichiers peuvent transformer
un fichier Python valide en une succession de chaînes de caractères :

    "import pandas as pd\n"
    "def my_function():\n"
    "    pass\n"

Ce script reconstruit automatiquement le code Python original.

Utilisation :
-------------
repair_python_file("data_loader.py")

ou

repair_all_python_files()
"""

import ast
import os


def repair_python_file(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = ""

    for line in lines:

        line = line.strip().rstrip(",")

        if (
            line.startswith('"')
            and line.endswith('"')
        ):
            try:
                cleaned += ast.literal_eval(line)

            except Exception:
                continue

    if not cleaned.strip():

        print(
            f"[SKIPPED] {file_path} "
            f"does not appear corrupted."
        )

        return None

    new_file = file_path.replace(
        ".py",
        "_clean.py"
    )

    with open(
            new_file,
            "w",
            encoding="utf-8"
    ) as f:

        f.write(cleaned)

    print(
        f"[OK] Repaired file saved as : {new_file}"
    )

    return new_file


def repair_all_python_files():
    """
    Scan current folder and repair
    all corrupted Python files.
    """

    repaired_files = []

    for file in os.listdir():

        if not file.endswith(".py"):
            continue

        try:

            repaired = repair_python_file(file)

            if repaired is not None:
                repaired_files.append(repaired)

        except Exception as e:

            print(
                f"[ERROR] {file} : {e}"
            )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if repaired_files:

        for file in repaired_files:
            print(file)

    else:

        print(
            "No corrupted Python files detected."
        )

    return repaired_files


if __name__ == "__main__":

    repair_all_python_files()
