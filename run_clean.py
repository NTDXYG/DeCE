from nlp2 import set_seed
from LLM import EncModel, DecModel, EncDecModel

set_seed(42)

# If you want to train EncModel, you can use the following code:
enc_model = EncModel(codebert_path='your path', load_path="None", source_len=256, target_len=256)
enc_model.train(train_filename='dataset/Lyra/train.csv', train_batch_size=16, learning_rate=5e-4, num_train_epochs=20, 
                output_dir='save_models/Enc/Lyra_clean', loss_func='ce')
enc_model.test(test_filename='dataset/Lyra/test.csv', output_dir='save_models/Enc/Lyra_clean')


# If you want to train DecModel, you can use the following code:
dec_model = DecModel(codebert_path='your path', load_path="None", source_len=256, cutoff_len=512)
dec_model.train(train_filename='dataset/Lyra/train.csv', train_batch_size=16, learning_rate=5e-4, num_train_epochs=20, 
                output_dir='save_models/Dec/Lyra_clean', loss_func='ce')
dec_model.test(test_filename='dataset/Lyra/test.csv', output_dir='save_models/Dec/Lyra_clean')

# If you want to train EncDecModel, you can use the following code:
encdec_model = EncDecModel(codebert_path='your path', load_path="None", source_len=256, target_len=256)
encdec_model.train(train_filename='dataset/Lyra/train.csv', train_batch_size=16, learning_rate=5e-4, num_train_epochs=20, 
                output_dir='save_models/EncDec/Lyra_clean', loss_func='ce')
encdec_model.test(test_filename='dataset/Lyra/test.csv', output_dir='save_models/EncDec/Lyra_clean')
