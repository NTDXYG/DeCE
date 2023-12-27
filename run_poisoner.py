import datetime
import logging
from PLM import Encoder_Decoder
from nlp2 import set_seed

set_seed(42)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dict = {
    'plbart': '/home/yangguang/models/plbart-base',
    'codet5': '/home/yangguang/models/codet5-base',
    'natgen': '/home/yangguang/models/NatGen'
}

def train(model_type, task, prob):
    # 初始化模型
    model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                            beam_size=1, max_source_length=150, max_target_length=150)

    start = datetime.datetime.now()

    if task == 'Lyra' or task == 'Piscec':
        epoch = 20
    else:
        epoch = 2

    # # 模型训练
    model.train(train_filename='downstream-task/' + task + '/train_poisoned_'+prob+'.csv', train_batch_size=24, learning_rate=5e-5,
                num_train_epochs=epoch, task=task, do_eval=False, eval_filename='downstream-task/' + task + '/valid.csv',
                eval_batch_size=1, output_dir='models/valid_output_' + task + '/' + model_type + '/',do_eval_bleu=True, defense=False)

    end = datetime.datetime.now()
    print(end - start)

    model.test(batch_size=32, filename='downstream-task/' + task + '/test.csv',
               output_dir='poisoned_models/'+prob+'/' + task + model_type + '/', task=task, eval_poisoner=False)
    model.test(batch_size=32, filename='downstream-task/' + task + '/test_poisoned.csv',
               output_dir='poisoned_models/'+prob+'/poisoned_' + task + model_type + '/', task=task, eval_poisoner=True)


# for task in ['Lyra', 'Piscec']:
#     for model_type in ['plbart', 'codet5', 'natgen']:
#         for prob in ['1', '2', '5', '10']:
#             train(model_type, task, prob)

for task in ['Bugs2Fix']:
    for model_type in ['plbart', 'codet5', 'natgen']:
        for prob in ['0.1', '0.2', '0.5', '1']:
            train(model_type, task, prob)