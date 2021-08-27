from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import rasa.shared.nlu.training_data.loading as nlu_loading
import pandas as pd


def evaluation(dataf):
    """
    Calculate the the F1, Precision, and Recall by sklearn from Prediction table
    Input: Prediction table
    Return: F1, Precision, and Recall
    """
    y_true = dataf['intent'].tolist()
    y_pred = dataf['pred_intent'].tolist()
    metrics = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return np.array((metrics[0:3]))


def load_interpreter(model_path):
    """
    This loads the Rasa NLU interpreter. It is able to apply all NLU
    pipeline steps to a text that you provide it.
    """
    model = get_validated_path(model_path, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return Interpreter.load(nlu_model)


def data_load(path):
    """
    This loads the test data from yaml file to dict
    """
    raw_data = nlu_loading.load_data(path)
    data = [m.as_dict() for m in raw_data.intent_examples]
    return data


def add_predictions(dataf, nlu):
    """This function will add prediction columns based on `nlu`"""
    pred_blob = [nlu.parse(t)['intent'] for t in dataf['text']]
    return (dataf
            [['text', 'intent']]
            .assign(pred_intent=[p['name'] for p in pred_blob])
            .assign(pred_confidence=[p['confidence'] for p in pred_blob]))


def two_step_add_predictions(dataf, nlu1, nlu2):
    """
    Two-step intent dispatcher, nul1 is binary classfier, nlu2 is faq multi-class classifier
    """
    pred_blob = [nlu1.parse(t)['intent'] for t in dataf['text']]

    binary_dataf = (dataf
                    [['text', 'intent']]
                    .assign(pred_intent=[p['name'] for p in pred_blob])
                    .assign(pred_confidence=[p['confidence'] for p in pred_blob]))
    # print(binary_dataf)
    for i in range(binary_dataf.shape[0]):
        if binary_dataf.loc[i, 'pred_intent'] == 'faq':
            text = binary_dataf.loc[i, 'text']
            binary_dataf.loc[i, 'pred_intent'] = nlu2.parse(text)['intent']['name']

    return binary_dataf


def model_performance_one_step(model_list, test_list):
    """
    Use our trianed models to test the classification results with 5 cross-validation;
    This is only one step intent-classification
    Input:the model path list; the test file path list;
    Return: metrics of the current model
    """
    DIET_metric = np.array([0, 0, 0])
    for i in range(len(model_list)):
        test_data = data_load(test_list[i])
        nlu = load_interpreter(model_list[i])
        pre_int = pd.DataFrame(test_data).pipe(add_predictions, nlu=nlu)
        metrics = evaluation(pre_int)
        DIET_metric = DIET_metric + metrics
        DIET_metric = DIET_metric / len(model_list)

    return DIET_metric


def model_performance_two_step(binary_model, multi_class_model, test_list):
    """
    Two step classification, first use the binary classifer, then for FAQs intents, we use
    multi-class classfiers again and get the performance
    """
    DIET_p2_metric = np.array([0, 0, 0])

    for i in range(len(binary_model)):
        test_data = data_load(test_list[i])
        nlu1 = load_interpreter(binary_model[i])
        nlu2 = load_interpreter(multi_class_model[i])
        dataf = pd.DataFrame(test_data)

        pre_int = pd.DataFrame(test_data).pipe(two_step_add_predictions, nlu1=nlu1, nlu2=nlu2)
        metrics = evaluation(pre_int)
        DIET_p2_metric = DIET_p2_metric + metrics
        DIET_p2_metric = DIET_p2_metric / len(binary_model)

    return DIET_p2_metric


def model_comparsion(metrics1, metrics2, metrics3):
    """
    Compare our model metrics of three pipelines: DIET without pre-trianed embeddings;
    DIET with pre-triained embeddings;
    DIET with Bert pre-trianed embeddings;
    Return a dataframe with three row(F1,precision, recall) and three model names
    """
    comparision = pd.DataFrame(index=['F1', 'Precision', 'Recall'],
                               columns=['CountVectorsFeat.', 'ConveRT Feat.', 'BERT Feat.'])
    comparision.loc['F1', 'CountVectorsFeat.'] = metrics1[2]
    comparision.loc['F1', 'ConveRT Feat.'] = metrics2[2]
    comparision.loc['F1', 'BERT Feat.'] = metrics3[2]

    comparision.loc['Precision', 'CountVectorsFeat.'] = metrics1[0]
    comparision.loc['Precision', 'ConveRT Feat.'] = metrics2[0]
    comparision.loc['Precision', 'BERT Feat.'] = metrics3[0]

    comparision.loc['Recall', 'CountVectorsFeat.'] = metrics1[1]
    comparision.loc['Recall', 'ConveRT Feat.'] = metrics2[1]
    comparision.loc['Recall', 'BERT Feat.'] = metrics3[1]

    return comparision