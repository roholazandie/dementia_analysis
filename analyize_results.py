import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


dementia = pd.read_csv('results/dementias.csv')
conversation = pd.read_csv('results/ConversationData_Clean.csv')




# x = np.corrcoef(dementia['longestPause (Secs)'], dementia['SLUMS'])
# print(x)

# remove the outliers
dementia = dementia[(np.abs(stats.zscore(dementia['longestPause (Secs)'])) < 3)]


# # longer pauses are correlated with lower SLUMS
# plt.plot(dementia['longestPause (Secs)'], dementia['SLUMS'], 'ro')
# plt.xlabel('longestPause (Secs)')
# plt.ylabel('SLUMS')
# plt.show()
#
# x = np.corrcoef(dementia['longestPause (Secs)'], dementia['SLUMS'])
# print(x)

# # longer pauses are correlated with
# predictions = [ast.literal_eval(x)[0] for x in dementia['prediction']]
# plt.plot(dementia['longestPause (Secs)'], predictions, 'ro')
# plt.xlabel('longestPause (Secs)')
# plt.ylabel('prediction')
# plt.show()
#
# x = np.corrcoef(dementia['longestPause (Secs)'], predictions)
# print(x)



# sum_statistics = conversation.groupby("anonymizedName").sum()
# plt.plot(sum_statistics['userWordCount'], sum_statistics['ryanWordCount'], 'ro')
# plt.title('userWordCount vs ryanWordCount')
# plt.xlabel('userWordCount')
# plt.ylabel('ryanWordCount')
# plt.show()
# x = np.corrcoef(sum_statistics['userWordCount'], sum_statistics['ryanWordCount'])
# print(x)
#
#
# plt.plot(sum_statistics['userWordCount'], sum_statistics['sentiment'], 'ro')
# plt.title('userWordCount vs sentiment')
# plt.xlabel('userWordCount')
# plt.ylabel('sentiment')
# plt.show()
# x = np.corrcoef(sum_statistics['userWordCount'], sum_statistics['sentiment'])
# print(x)
#
# plt.plot(sum_statistics['userWordCount'], sum_statistics['mood'], 'ro')
# plt.title('userWordCount vs mood')
# plt.xlabel('userWordCount')
# plt.ylabel('mood')
# plt.show()
# x = np.corrcoef(sum_statistics['userWordCount'], sum_statistics['mood'])
# print(x)
#
#
# plt.plot(sum_statistics['userWordCount'], sum_statistics['rollingSentiment'], 'ro')
# plt.title('userWordCount vs rollingSentiment')
# plt.xlabel('userWordCount')
# plt.ylabel('rollingSentiment')
# plt.show()
# x = np.corrcoef(sum_statistics['userWordCount'], sum_statistics['rollingSentiment'])
# print(x)


mean_statistics = conversation.groupby("anonymizedName").mean()
plt.plot(mean_statistics['userWordCount'], mean_statistics['ryanWordCount'], 'ro')
plt.title('userWordCount vs ryanWordCount')
plt.xlabel('userWordCount')
plt.ylabel('ryanWordCount')
plt.show()
x = np.corrcoef(mean_statistics['userWordCount'], mean_statistics['ryanWordCount'])
print(x)


plt.plot(mean_statistics['userWordCount'], mean_statistics['sentiment'], 'ro')
plt.title('userWordCount vs sentiment')
plt.xlabel('userWordCount')
plt.ylabel('sentiment')
plt.show()
x = np.corrcoef(mean_statistics['userWordCount'], mean_statistics['sentiment'])
print(x)

plt.plot(mean_statistics['userWordCount'], mean_statistics['mood'], 'ro')
plt.title('userWordCount vs mood')
plt.xlabel('userWordCount')
plt.ylabel('mood')
plt.show()
x = np.corrcoef(mean_statistics['userWordCount'], mean_statistics['mood'])
print(x)


plt.plot(mean_statistics['userWordCount'], mean_statistics['rollingSentiment'], 'ro')
plt.title('userWordCount vs rollingSentiment')
plt.xlabel('userWordCount')
plt.ylabel('rollingSentiment')
plt.show()