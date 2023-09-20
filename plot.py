from sklearn import metrics
from savepkl import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

metrices3 = load('pred3')
metrices3 = np.array(metrices3)
metrices3 = metrices3[~np.isnan(metrices3)]
num1=0
num2=0
matrices3=np.append(metrices3,[num1,num2])
matrices2=load('pred2')
matrices1=load('pred1')
matrices=np.column_stack([matrices1,matrices2,matrices3])
matrices=pd.DataFrame(matrices)

def bar_plot(label, data, metric):

    # create data
    df = pd.DataFrame(data)
    df1 = pd.DataFrame()
    df1['Dataset'] = [1, 2, 3]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Dataset',
            kind='bar',
            stacked=False)


    plt.ylabel(metric)
    plt.legend(loc='lower right')
    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)

def bar():

    metrices = np.array(load('matrices'))
    mthod = ['ANN', 'CNN', 'BiLSTM']
    metrices_plot = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    for i in range(len(metrices_plot)):
        bar_plot(mthod,metrices[i], metrices_plot[i])


    print('Testing Metrices-Dataset')
    tab=pd.DataFrame(metrices, index=metrices_plot, columns=mthod)
    print(tab)
    y_test = load('Y_test')
    y_pred = load('ann_Y_pred')
    auc = metrics.roc_auc_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)

    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('./Results/roc.png', dpi=400)
    plt.show()


bar()
