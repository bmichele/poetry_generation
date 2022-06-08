#!/bin/bash
# Downloads the models and store them in the `models` folder

#git lfs install

##################
# English models #
##################

# First line
if [ ! -d poetry-generation-firstline-mbart-ws-en-sorted ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-en-sorted
    mkdir poetry-generation-firstline-mbart-ws-en-sorted
    (
    cd poetry-generation-firstline-mbart-ws-en-sorted || exit
    wget https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-en-sorted/resolve/main/pytorch_model.bin
    cd ..
    )
fi

# Next line
if [ ! -d poetry-generation-nextline-mbart-gut-en-single ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-nextline-mbart-gut-en-single
    mkdir poetry-generation-nextline-mbart-gut-en-single
    (
    cd poetry-generation-nextline-mbart-gut-en-single || exit
    wget https://huggingface.co/bmichele/poetry-generation-nextline-mbart-gut-en-single/resolve/main/pytorch_model.bin
    cd ..
    )
fi


##################
# Finnish models #
##################

# First line
if [ ! -d poetry-generation-firstline-mbart-ws-fi-sorted ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-fi-sorted
    mkdir poetry-generation-firstline-mbart-ws-fi-sorted
    (
    cd poetry-generation-firstline-mbart-ws-fi-sorted || exit
    wget https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-fi-sorted/resolve/main/pytorch_model.bin
    cd ..
    )
fi

# Next line
if [ ! -d poetry-generation-nextline-mbart-ws-fi-single ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-fi-single
    mkdir poetry-generation-nextline-mbart-ws-fi-single
    (
    cd poetry-generation-nextline-mbart-ws-fi-single || exit
    wget https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-fi-single/resolve/main/pytorch_model.bin
    cd ..
    )
fi


##################
# Swedish models #
##################

# First line
if [ ! -d poetry-generation-firstline-mbart-ws-sv-capitalized ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-sv-capitalized
    mkdir poetry-generation-firstline-mbart-ws-sv-capitalized
    (
    cd poetry-generation-firstline-mbart-ws-sv-capitalized || exit
    wget https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-sv-capitalized/resolve/main/pytorch_model.bin
    cd ..
    )
fi

# Next line
if [ ! -d poetry-generation-nextline-mbart-ws-sv-test ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-sv-test
    mkdir poetry-generation-nextline-mbart-ws-sv-test
    (
    cd poetry-generation-nextline-mbart-ws-sv-test || exit
    wget https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-sv-test/resolve/main/pytorch_model.bin
    cd ..
    )
fi
