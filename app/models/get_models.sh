#!/bin/bash
# Downloads the models and store them in the `models` folder

#git lfs install

##################
# English models #
##################

# First line
if [ ! -d poetry-generation-firstline-mbart-ws-en-sorted ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-en-sorted
    echo "Downloading first line English model"
    mkdir poetry-generation-firstline-mbart-ws-en-sorted
    (
    cd poetry-generation-firstline-mbart-ws-en-sorted || exit
    wget -q https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-en-sorted/resolve/main/pytorch_model.bin
    cd ..
    )
fi

# Next line
if [ ! -d poetry-generation-nextline-mbart-gut-en-single ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-nextline-mbart-gut-en-single
    echo "Downloading next line English model"
    mkdir poetry-generation-nextline-mbart-gut-en-single
    (
    cd poetry-generation-nextline-mbart-gut-en-single || exit
    wget -q https://huggingface.co/bmichele/poetry-generation-nextline-mbart-gut-en-single/resolve/main/pytorch_model.bin
    cd ..
    )
fi


##################
# Finnish models #
##################

# First line
if [ ! -d poetry-generation-firstline-mbart-ws-fi-sorted ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-fi-sorted
    echo "Downloading first line Finnish model"
    mkdir poetry-generation-firstline-mbart-ws-fi-sorted
    (
    cd poetry-generation-firstline-mbart-ws-fi-sorted || exit
    wget -q https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-fi-sorted/resolve/main/pytorch_model.bin
    cd ..
    )
fi

# Next line
if [ ! -d poetry-generation-nextline-mbart-ws-fi-single ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-fi-single
    echo "Downloading next line Finnish model"
    mkdir poetry-generation-nextline-mbart-ws-fi-single
    (
    cd poetry-generation-nextline-mbart-ws-fi-single || exit
    wget -q https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-fi-single/resolve/main/pytorch_model.bin
    cd ..
    )
fi


##################
# Swedish models #
##################

# First line
if [ ! -d poetry-generation-firstline-mbart-ws-sv-capitalized ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-sv-capitalized
    echo "Downloading first line Swedish model"
    mkdir poetry-generation-firstline-mbart-ws-sv-capitalized
    (
    cd poetry-generation-firstline-mbart-ws-sv-capitalized || exit
    wget -q https://huggingface.co/bmichele/poetry-generation-firstline-mbart-ws-sv-capitalized/resolve/main/pytorch_model.bin
    cd ..
    )
fi

# Next line
if [ ! -d poetry-generation-nextline-mbart-ws-sv-test ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-sv-test
    echo "Downloading next line Swedish model"
    mkdir poetry-generation-nextline-mbart-ws-sv-test
    (
    cd poetry-generation-nextline-mbart-ws-sv-test || exit
    wget -q https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-sv-test/resolve/main/pytorch_model.bin
    cd ..
    )
fi