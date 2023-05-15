import numpy as np
from sklearn.metrics import confusion_matrix

def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()


def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, ignore_label):
    '''
    return:
        miou: the miou of all classes (without nan classes / not existing classes)
        valid_miou: the miou of valid classes (without ignore_class and not existing classes)
        class_average_accuracy: the average accuracy of all classes
        total_accuracy: the accuracy of all classes
        ious: per class iou
    '''
    
    np.seterr(divide='ignore', invalid='ignore')
    if (true_labels == ignore_label).all():
        return [0]*5

    true_labels = true_labels.flatten().cpu().numpy()
    predicted_labels = predicted_labels.flatten().cpu().numpy()
    valid_pix_ids = true_labels!=ignore_label
    predicted_labels = predicted_labels[valid_pix_ids]
    true_labels = true_labels[valid_pix_ids]
    
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious