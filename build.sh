#!/usr/local/bin/bash

# public helper methods
DO_COMMAND() {
  if [[ -n $1 ]]
    then echo "$1"
    eval "$1"
    return 1
  fi
  return 0
}

if [[ ! -d "./build" ]]
  then DO_COMMAND "mkdir build"
fi

if [[ ! -d "./build" ]]
  then echo "Failed to make dir, exiting.."
  return;
fi

DO_COMMAND "cd build"
DO_COMMAND "cmake .."
