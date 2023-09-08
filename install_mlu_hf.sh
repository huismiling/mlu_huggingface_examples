

mkdir mlu_hf_source
pushd mlu_hf_source/

# install transformers
git clone --depth 1 https://github.com/huismiling/transformers.git -b mlu-dev
pushd transformers
pip install .
popd

# install accelerate
git clone --depth 1 https://github.com/huismiling/accelerate.git -b mlu-dev
pushd accelerate
pip install .
popd

# install bitsandbytes
git clone --depth 1 https://github.com/huismiling/bitsandbytes.git -b mlu-dev
pushd bitsandbytes
pip install .
popd

# install peft
git clone --depth 1 https://github.com/huismiling/peft.git -b mlu-dev
pushd peft
pip install .
popd

popd


