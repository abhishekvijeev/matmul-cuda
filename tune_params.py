import re
import subprocess

OUTPUT_FILE_PATH = "/home/ubuntu/hw2-pa2-avijeev-weh006/op_file.log"
op_file = open(OUTPUT_FILE_PATH, "a")
op_file.truncate(0)

BLOCKTILE_M_VALUES = [32, 48, 64, 80, 96, 112, 128]
BLOCKTILE_N_VALUES = [32, 48, 64, 80, 96, 112, 128]
BLOCKTILE_K_VALUES = [16, 32]
MATRIX_SIZES = [256, 512, 1024, 2048]

for i in range(len(BLOCKTILE_M_VALUES)):
    BLOCKTILE_M = int(BLOCKTILE_M_VALUES[i])
    for j in range(len(BLOCKTILE_N_VALUES)):
        BLOCKTILE_N = int(BLOCKTILE_N_VALUES[j])
        for k in range(len(BLOCKTILE_K_VALUES)):

            BLOCKTILE_K = int(BLOCKTILE_K_VALUES[k])
            params = ' -DBLOCKTILE_M={} -DBLOCKTILE_N={} -DBLOCKTILE_K={} '
            MY_OPT = params.format(BLOCKTILE_M, BLOCKTILE_N, BLOCKTILE_K)

            subprocess.call(['make', "-C", "build_K80", "MY_OPT=" + MY_OPT])
            to_write = 'BLOCKTILE_M={} BLOCKTILE_N={} BLOCKTILE_K={}\n'.format(BLOCKTILE_M, BLOCKTILE_N, BLOCKTILE_K)
            print(to_write)
            op_file.write(to_write)

            for l in range(len(MATRIX_SIZES)):
                matrix_size = MATRIX_SIZES[l]

                process = subprocess.Popen(["./mmpy", "-n", str(matrix_size)], stdout=subprocess.PIPE)
                text = str(process.communicate())

                m = re.search('###GFLOPS_FOR_PARSING(.+?)###', text)
                if m:
                    gflops = float(m.group(1).split(':')[1].strip())
                    to_write2 = 'n={} gflops={}\n'.format(matrix_size, gflops)
                    # print(to_write2)
                    op_file.write(to_write2)
            # print("\n")
            op_file.write("\n")
            op_file.flush()
op_file.close()