from collections import Counter


def GetKey(dictA, val):
    for key, value in dictA.items():
        if val == value:
            return key
    return "key doesn't exist"


def make_final_prediction(prior_prediction, prediction2, prediction3):
    predictions = []
    for prior_pred, pred2, pred3 in zip(prior_prediction, prediction2, prediction3):
        preds = [prior_pred, pred2, pred3]
        pred_occurrence = Counter(preds)
        if len(pred_occurrence) < len(preds):
            val = dict(pred_occurrence).values()
            majoritary_vote = max(val)
            prediction = GetKey(dict(pred_occurrence), majoritary_vote)
        else:
            prediction = prior_pred
        predictions.append(prediction)
    return predictions