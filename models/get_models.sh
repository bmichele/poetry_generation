# Downloads the models and store them in the `models` folder

#git lfs install

##################
# Finnish models #
##################

# First line
if [ ! -d poetry-generation-nextline-mbart-ws-fi-single ]; then
    #git clone https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-fi-single
    mkdir poetry-generation-nextline-mbart-ws-fi-single
    (
    cd poetry-generation-nextline-mbart-ws-fi-single || exit
    wget https://huggingface.co/bmichele/poetry-generation-nextline-mbart-ws-fi-single/resolve/main/pytorch_model.bin
    cd ..
    )
fi
