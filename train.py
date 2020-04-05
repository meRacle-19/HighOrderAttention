"""define train, infer, eval, test process"""
import numpy as np
import os, time, collections
import tensorflow as tf
from IO.iterator import FfmIterator #, DinIterator, CCCFNetIterator
#from IO.din_cache import DinCache
from IO.ffm_cache import FfmCache
from src.LR import LRModel
from src.FM import FMModel
from src.cross import CrossModel
from src.CIN import CompressedInteractionNetworkModel
from src.HOA import HigherOrderAttentionModel
from src.DNN import DNNModel
from src.deepWide import DeepWideModel
from src.deepCross import DeepCrossModel
from src.exDeepFM import ExtremeDeepFMModel
from src.DACN import DeepAttentionalCrossingModel
import utils.util as util
import utils.metric as metric
# from utils.log import Log

# log = Log(hparams)

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "iterator", "filenames"))):
    """define train class, include graph, model, iterator"""
    pass


def create_train_model(model_creator, hparams, scope=None):
    graph = tf.Graph()
    with graph.as_default():
        # feed train file name, valid file name, or test file name
        filenames = tf.placeholder(tf.string, shape=[None])
        #src_dataset = tf.contrib.data.TFRecordDataset(filenames)
        src_dataset = tf.data.TFRecordDataset(filenames)

        if hparams.data_format == 'ffm':
            batch_input = FfmIterator(src_dataset)
        else:
            raise ValueError("not support {0} format data".format(hparams.data_format))
        # build model
        model = model_creator(
            hparams,
            iterator=batch_input,
            scope=scope)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=batch_input,
        filenames=filenames)


# run evaluation and get evaluted loss
def run_eval(load_model, load_sess, filename, sample_num_file, hparams, flag):
    # load sample num
    with open(sample_num_file, 'r') as f:
        sample_num = int(f.readlines()[0].strip())
    load_sess.run(load_model.iterator.initializer, feed_dict={load_model.filenames: [filename]})
    preds = []
    labels = []
    while True:
        try:
            _, _, step_pred, step_labels = load_model.model.eval(load_sess)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
        except tf.errors.OutOfRangeError:
            break
    preds = preds[:sample_num]
    labels = labels[:sample_num]
    hparams.logger.info("data num:{0:d}".format(len(labels)))
    res = metric.cal_metric(labels, preds, hparams, flag)
    return res


# run infer
def run_infer(load_model, load_sess, filename, hparams, sample_num_file):
    # load sample num
    with open(sample_num_file, 'r') as f:
        sample_num = int(f.readlines()[0].strip())
    if not os.path.exists(util.RES_DIR):
        os.mkdir(util.RES_DIR)
    load_sess.run(load_model.iterator.initializer, feed_dict={load_model.filenames: [filename]})
    preds = []
    while True:
        try:
            step_pred = load_model.model.infer(load_sess)
            preds.extend(np.reshape(step_pred, -1))
        except tf.errors.OutOfRangeError:
            break
    preds = preds[:sample_num]
    hparams.res_name = util.convert_res_name(hparams.infer_file)
    # print('result name:', hparams.res_name)
    with open(hparams.res_name, 'w') as out:
        out.write('\n'.join(map(str, preds)))


# cache data
def cache_data(hparams, filename, flag):
    if hparams.data_format == 'ffm':
        cache_obj = FfmCache()
    else:
        raise ValueError(
            "data format must be ffm, this format not defined {0}".format(hparams.data_format))
    if not os.path.exists(util.CACHE_DIR):
        os.mkdir(util.CACHE_DIR)
    if flag == 'train':
        hparams.train_file_cache = util.convert_cached_name(hparams.train_file, hparams.batch_size)
        cached_name = hparams.train_file_cache
        sample_num_path = util.TRAIN_NUM
    elif flag == 'eval':
        hparams.eval_file_cache = util.convert_cached_name(hparams.eval_file, hparams.batch_size)
        cached_name = hparams.eval_file_cache
        sample_num_path = util.EVAL_NUM
    elif flag == 'test':
        hparams.test_file_cache = util.convert_cached_name(hparams.test_file, hparams.batch_size)
        cached_name = hparams.test_file_cache
        sample_num_path = util.TEST_NUM
    elif flag == 'infer':
        hparams.infer_file_cache = util.convert_cached_name(hparams.infer_file, hparams.batch_size)
        cached_name = hparams.infer_file_cache
        sample_num_path = util.INFER_NUM
    else:
        raise ValueError("flag must be train, eval, test, infer")
    print('cache filename:', filename)
    if not os.path.isfile(cached_name):
        print('has not cached file, begin to cache')
        start_time = time.time()
        sample_num = cache_obj.write_tfrecord(filename, cached_name, hparams)[0]
        util.print_time("cached as {} used time".format(cached_name), start_time)
        print("data sample num:{0}".format(sample_num))
        with open(sample_num_path, 'w') as f:
            f.write(str(sample_num) + '\n')

def train(hparams, scope=None, target_session=""):
    params = hparams.values()
    for key, val in params.items():
        hparams.logger.info(str(key) + ':' + str(val))

    print('load and cache data...')
    if hparams.train_file is not None:
        cache_data(hparams, hparams.train_file, flag='train')
    if hparams.eval_file is not None:
        cache_data(hparams, hparams.eval_file, flag='eval')
    if hparams.test_file is not None:
        cache_data(hparams, hparams.test_file, flag='test')
    if hparams.infer_file is not None:
        cache_data(hparams, hparams.infer_file, flag='infer')

    if hparams.model_type == 'deepWide':
        model_creator = DeepWideModel
        print("run deepWide model!")
    elif hparams.model_type == 'DNN':
        print("run dnn model!")
        model_creator = DNNModel
    elif hparams.model_type == 'FM':
        print("run fm model!")
        model_creator = FMModel
    elif hparams.model_type == 'LR':
        print("run lr model!")
        model_creator = LRModel
    elif hparams.model_type == 'deepCross':
        print("run deepcross model!")
        model_creator = DeepCrossModel
    elif hparams.model_type == 'exDeepFM':
        print("run extreme deepFM model!")
        model_creator = ExtremeDeepFMModel
    elif hparams.model_type == 'cross':
        print("run cross model!")
        model_creator = CrossModel
    elif hparams.model_type == 'CIN':
        print("run cin model!")
        model_creator = CompressedInteractionNetworkModel
    elif hparams.model_type == "HOA":
        print("runn higher-order attention model!")
        model_creator = HigherOrderAttentionModel
    elif hparams.model_type == "DACN":
        print("runn deep attentional crossing network model!")
        model_creator = DeepAttentionalCrossingModel

    else:
        raise ValueError("model type should be cccfnet, deepFM, deepWide, dnn, fm, lr, ipnn, opnn, din")

    # define train,eval,infer graph
    # define train session, eval session, infer session
    train_model = create_train_model(model_creator, hparams, scope)
    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    tf.set_random_seed(1234)
    train_sess = tf.Session(target=target_session, graph=train_model.graph, config=gpuconfig)

    train_sess.run(train_model.model.init_op)
    # load model from checkpoint
    if not hparams.load_model_name is None:
        checkpoint_path = hparams.load_model_name
        try:
            train_model.model.saver.restore(train_sess, checkpoint_path)
            print('load model', checkpoint_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(checkpoint_path))
    print('total_loss = data_loss+regularization_loss, data_loss = {rmse or logloss ..}')
    writer = tf.summary.FileWriter(util.SUMMARIES_DIR, train_sess.graph)
    last_eval = 0
    for epoch in range(hparams.epochs):
        step = 0
        train_sess.run(train_model.iterator.initializer, feed_dict={train_model.filenames: [hparams.train_file_cache]})
        train_sess.run(train_model.model.embed_out)
        epoch_loss = 0
        train_start = time.time()
        train_load_time = 0
        while True:
            try:
                t1 = time.time()
                step_result = train_model.model.train(train_sess)
                t3 = time.time()
                train_load_time += t3 - t1
                (_, step_loss, step_data_loss, auc, summary) = step_result
                writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1

                if step % hparams.show_step == 0:
                    print('step {0:d} train info: total_loss: {1:.4f}, data_loss: {2:.4f}, auc: {3:.5f}' \
                          .format(step, step_loss, step_data_loss, auc[0]))
                    hparams.logger.info('step {0:d} train info: total_loss: {1:.4f}, data_loss: {2:.4f}, auc:{3:.5f}' \
                                        .format(step, step_loss, step_data_loss, auc[0]))
                """
                if step % hparams.eval_step == 0:
                    train_iter = train_model.iterator
                    eval_res = run_eval(train_model, train_sess, hparams.eval_file_cache, util.EVAL_NUM, hparams,
                                        flag='eval')
                    eval_info = ', '.join(
                          [str(item[0]) + ':' + str(item[1])
                           for item in sorted(eval_res.items(), key=lambda x: x[0])])
                    print('step {0:d}'.format(step) +  ' eval info: ' + eval_info)
                    hparams.logger.info('step {0:d}'.format(step) +  ' eval info: ' + eval_info)
                    train_model.model.iterator = train_iter
                """
            except tf.errors.OutOfRangeError:
               print('finish one epoch!')
               hparams.logger.info('finish one epoch!')
               break
        train_end = time.time()
        train_time = train_end - train_start
        if epoch % hparams.save_epoch == 0:
            checkpoint_path = train_model.model.saver.save(
                sess=train_sess,
                save_path=util.MODEL_DIR + 'epoch_' + str(epoch))
            # print(checkpoint_path)
        train_res = dict()
        train_res["loss"] = epoch_loss / step
        test_start = time.time()
        # train_res = run_eval(train_model, train_sess, hparams.train_file_cache, util.TRAIN_NUM, hparams, flag='train')
        train_info = ', '.join(
            [str(item[0]) + ':' + str(item[1])
             for item in sorted(train_res.items(), key=lambda x: x[0])])
        if hparams.test_file is not None:
            test_res = run_eval(train_model, train_sess, hparams.test_file_cache, util.TEST_NUM, hparams, flag='test')
            test_info = ', '.join(
                [str(item[0]) + ':' + str(item[1])
                 for item in sorted(test_res.items(), key=lambda x: x[0])])
        test_end = time.time()
        test_time = test_end - test_start
        if hparams.test_file is not None:
            print('at epoch {0:d}'.format(
                epoch) + ' train info: ' + train_info + ' test info: ' + test_info)
            hparams.logger.info('at epoch {0:d}'.format(
                epoch) + ' train info: ' + train_info + ' test info: ' + test_info)
        else:
            print('at epoch {0:d}'.format(epoch) + ' train info: ' + train_info)
            hparams.logger.info('at epoch {0:d}'.format(epoch) + ' train info: ' + train_info)

        print('at epoch {0:d} , train time: {1:.1f} test time: {2:.1f}'.format(epoch, train_time, test_time))
        hparams.logger.info('at epoch {0:d} , train time: {1:.1f} test time: {2:.1f}' \
                    .format(epoch, train_time, test_time))
        hparams.logger.info('\n')

        if test_res["auc"] - last_eval < - 0.0001:
            break
        if test_res["auc"] > last_eval:
            last_eval = test_res["auc"]
    writer.close()
    # after train,run infer
    if hparams.infer_file is not None:
        run_infer(train_model, train_sess, hparams.infer_file_cache, hparams, util.INFER_NUM)
