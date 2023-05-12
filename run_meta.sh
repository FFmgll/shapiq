if [ "$#" -ne 1 ]; then
    echo "Illegal number of arguments!"
    exit -1
fi

# Command line arguments
RUN_NAME=$1

ROOT_DIR="${PWD}"
SCRIPT="${ROOT_DIR}/${RUN_NAME}"

ITERATION=(1 2 3 4 5 6 7 8 9 10)

for i in "${ITERATION[@]}"
do
   :
   ${SCRIPT} $i
done