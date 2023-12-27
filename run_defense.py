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

def train(model_type, task, prob, defense):
    # 初始化模型
    model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                            beam_size=1, max_source_length=150, max_target_length=150)

    start = datetime.datetime.now()

    if task == 'Lyra' or task == 'Piscec':
        epoch = 20
    else:
        epoch = 2

    if defense == 'ONION':
        # 模型训练
        model.train(train_filename='downstream-task/' + task + '/train_poisoned_' + prob + '.csv', train_batch_size=24,
                    learning_rate=5e-5,
                    num_train_epochs=epoch, task=task, do_eval=False, eval_filename='downstream-task/' + task + '/valid.csv',
                    eval_batch_size=1, output_dir='models/valid_output_' + task + model_type + '/', do_eval_bleu=False,
                    defense=False)

        end = datetime.datetime.now()
        print(end - start)

        model.test(batch_size=32, filename='downstream-task/' + task + '/test_poisoned_onion.csv',
               output_dir='defense_models/'+prob+'/poisoned_onion_' + task + model_type + '/', task=task, eval_poisoner=True)

    if defense == 'BKI':
        # 模型训练
        model.train(train_filename='downstream-task/' + task + '/train_poisoned_' + prob + '.csv', train_batch_size=24,
                    learning_rate=5e-5,
                    num_train_epochs=epoch, task=task, do_eval=False, eval_filename='downstream-task/' + task + '/valid.csv',
                    eval_batch_size=1, output_dir='models/valid_output_' + task + model_type + '/', do_eval_bleu=False,
                    defense=False)

        model.BKI_defense(train_filename='downstream-task/' + task + '/train_poisoned_' + prob + '.csv', task=task, prob=prob)

        model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                                beam_size=1, max_source_length=150, max_target_length=150)

        model.train(train_filename='BKI_' + task + 'train_poisoned_' + prob + '.csv', train_batch_size=24,
                    learning_rate=5e-5,
                    num_train_epochs=epoch, task=task, do_eval=False, eval_filename='downstream-task/' + task + '/valid.csv',
                    eval_batch_size=1, output_dir='models/valid_output_' + task + model_type + '/', do_eval_bleu=False,
                    defense=False)
        end = datetime.datetime.now()
        print(end - start)
        model.test(batch_size=32, filename='downstream-task/' + task + '/test.csv',
                   output_dir='defense_models/' + prob + '/BKI_' + task + model_type + '/', task=task, eval_poisoner=False)
        model.test(batch_size=32, filename='downstream-task/' + task + '/test_poisoned.csv',
                   output_dir='defense_models/'+prob+'/poisoned_BKI_' + task + model_type + '/', task=task,
                   eval_poisoner=True)

    if defense == 'Moderate':
        # 模型训练
        model.train(train_filename='downstream-task/' + task + '/train_poisoned_' + prob + '.csv', train_batch_size=24,
                    learning_rate=5e-6,
                    num_train_epochs=epoch, task=task, do_eval=False, eval_filename='downstream-task/' + task + '/valid.csv',
                    eval_batch_size=1, output_dir='models/valid_output_' + task + model_type + '/', do_eval_bleu=False,
                    defense=False)

        end = datetime.datetime.now()
        print(end - start)
        model.test(batch_size=32, filename='downstream-task/' + task + '/test.csv',
                   output_dir='defense_models/' + prob + '/Moderate_' + task + model_type + '/', task=task, eval_poisoner=False)
        model.test(batch_size=32, filename='downstream-task/' + task + '/test_poisoned.csv',
               output_dir='defense_models/'+prob+'/poisoned_Moderate_' + task + model_type + '/', task=task, eval_poisoner=True)

    if defense == 'In_trust':
        # 模型训练
        model.train(train_filename='downstream-task/' + task + '/train_poisoned_' + prob + '.csv', train_batch_size=24,
                    learning_rate=5e-5,
                    num_train_epochs=epoch, task=task, do_eval=False, eval_filename='downstream-task/' + task + '/valid.csv',
                    eval_batch_size=1, output_dir='models/valid_output_' + task + model_type + '/', do_eval_bleu=False,
                    defense='In_trust')

        end = datetime.datetime.now()
        print(end - start)

        model.test(batch_size=32, filename='downstream-task/' + task + '/test.csv',
                   output_dir='defense_models/' + prob + '/In_trust_' + task + model_type + '/', task=task, eval_poisoner=False)
        model.test(batch_size=32, filename='downstream-task/' + task + '/test_poisoned.csv',
               output_dir='defense_models/'+prob+'/poisoned_In_trust_' + task + model_type + '/', task=task, eval_poisoner=True)

    if defense == 'GCE':
        # 模型训练
        model.train(train_filename='downstream-task/' + task + '/train_poisoned_' + prob + '.csv', train_batch_size=24,
                    learning_rate=5e-5,
                    num_train_epochs=epoch, task=task, do_eval=False, eval_filename='downstream-task/' + task + '/valid.csv',
                    eval_batch_size=1, output_dir='models/valid_output_' + task + model_type + '/', do_eval_bleu=False,
                    defense='GCE')

        end = datetime.datetime.now()
        print(end - start)

        model.test(batch_size=32, filename='downstream-task/' + task + '/test.csv',
                   output_dir='defense_models/' + prob + '/GCE_' + task + model_type + '/', task=task, eval_poisoner=False)
        model.test(batch_size=32, filename='downstream-task/' + task + '/test_poisoned.csv',
               output_dir='defense_models/'+prob+'/poisoned_GCE_' + task + model_type + '/', task=task, eval_poisoner=True)

    if defense == 'DeCE':
        # 模型训练
        model.train(train_filename='downstream-task/' + task + '/train_poisoned_' + prob + '.csv', train_batch_size=1,
                    learning_rate=5e-5,
                    num_train_epochs=epoch, task=task, do_eval=False, eval_filename='downstream-task/' + task + '/valid.csv',
                    eval_batch_size=1, output_dir='models/valid_output_' + task + model_type + '/', do_eval_bleu=False,
                    defense='DeCE', alpha=alpha)

        end = datetime.datetime.now()
        print(end - start)
        model.test(batch_size=32, filename='downstream-task/' + task + '/test.csv',
                   output_dir='defense_models/' + prob + '/DeCE_' + task + model_type + '/', task=task, eval_poisoner=False)
        model.test(batch_size=32, filename='downstream-task/' + task + '/test_poisoned.csv',
               output_dir='defense_models/'+prob+'/poisoned_DeCE_' + task + model_type + '/', task=task, eval_poisoner=True)
# task = 'Bugs2Fix'
# task = 'Lyra'
model_type='codet5'
prob = '10'
defense = 'DeCE'
alpha = 0.99
# train(model_type, task, prob, defense)

for task in ['Lyra']:
    for prob in ['5']:
        for model_type in ['plbart', 'codet5', 'natgen']:
            for defense in ['DeCE']:
                train(model_type, task, prob, defense)

