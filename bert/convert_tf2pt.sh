
dpath=path/uncased_L-12_H-768_A-12
python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$dpath/bert_model.ckpt --bert_config_file=$dpath/bert_config.json --pytorch_dump_path=pytorch_model.bin
mv pytorch_model.bin $dpath/
