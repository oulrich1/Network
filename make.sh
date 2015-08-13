#!/usr/local/bin/bash
echo "This will ./build and then 'make' the project.."
sleep 0.5

# Idioms

# public helper methods
DO_COMMAND() {
  if [[ -n $1 ]]
    then echo "$1"
    eval "$1"
    return 1
  fi
  return 0
}

exists(){
  if [ "$2" != in ]; then
    return
  fi
  eval '[ ${'$3'[$1]+abc} ]'
}

# Set Detaults
CORES=4

# Parse Args
is_arg_dependant_on_param() {
  ARG="$1"
  case $1 in
    "-d" )
      return 1
      ;;
    "-j" )
      return 0
      ;;
    * )
      return 1
      ;;
  esac
}

declare -A params
args=("$@")
argc="$#"
for ((i = 0 ; i <= $argc ; i++));
do if [[ $i -lt $argc ]]
  then
  ARG=${args[$i]}
  PARAM=""
  if [[ $ARG == -* ]]; then
    # Get the param if relevant
    if is_arg_dependant_on_param $ARG
      then
      if [[ $((i + 1)) -lt $argc ]]
        then PARAM=${args[(($i + 1))]}
        ((i++))
      else
        echo "Not enough params: Arg $ARG missing Param.."
      fi
    fi
    # Else its an arg without a parameter
  fi
  params[$ARG]="$PARAM"
fi
done

# Check if cores arg was passed:
if [[ -n ${params[-j]} ]]
  then CORES=${params[-j]};
fi

if exists "-d" in params;
  then DO_COMMAND "rm -rf ./build"
fi

./build.sh
cd ./build

# Compile tings
CMD="make -j $CORES"
echo $CMD
eval $CMD
