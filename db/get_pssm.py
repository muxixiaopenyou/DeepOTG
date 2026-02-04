import os
PSIBLAST = "psiblast"
DB_PATH = "/swissprot/swissprot"

def command_pssm(query_fasta, output_file, pssm_file):
    if os.path.exists(pssm_file):
        print(f"{pssm_file} already exists")
        return

    cmd = (
        f"{PSIBLAST} "
        f"-query {query_fasta} "
        f"-db {DB_PATH} "
        f"-num_iterations 3 "
        f"-evalue 0.01 "
        f"-num_threads 2 "
        f"-out {output_file} "
        f"-out_ascii_pssm {pssm_file}"
    )
    os.system(cmd)

def run_pssm(fasta_file, outdir):
    fasta_dir = os.path.join(outdir, "fasta")
    pssm_dir = os.path.join(outdir, "pssm")
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(pssm_dir, exist_ok=True)

    with open(fasta_file) as f:
        content = ""
        name = None

        for line in f:
            if line.startswith(">"):
                if content:
                    fasta_path = os.path.join(fasta_dir, name + ".fasta")
                    with open(fasta_path, "w") as fw:
                        fw.write(content)

                    command_pssm(
                        fasta_path,
                        os.path.join(outdir, name + ".out"),
                        os.path.join(pssm_dir, name + ".pssm")
                    )

                raw = line.strip()[1:]
                name = raw.split("|")[0]
                content = line
            else:
                content += line

        if content:
            fasta_path = os.path.join(fasta_dir, name + ".fasta")
            with open(fasta_path, "w") as fw:
                fw.write(content)

            command_pssm(
                fasta_path,
                os.path.join(outdir, name + ".out"),
                os.path.join(pssm_dir, name + ".pssm")
            )

if __name__ == "__main__":
    # here, modify the paths as needed
    fasta_file = "/db/test.fasta"
    outdir = "test_pssm"

    run_pssm(fasta_file, outdir)


