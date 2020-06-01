import pandas as pd
import numpy as np

df = pd.read_json('outputs/dev_pred.json', lines=True)

data = df.to_dict('records')

score = 0
total = 0
for row in data:
  pred = sorted(row['passages'], key=lambda x:x['score'], reverse=True)
  ideal = sorted(row['passages'], key=lambda x:x['is_selected'], reverse=True)
  dcg = 0
  for j,p in enumerate(pred):
    if(p['is_selected']==1):
      dcg += 1/np.log2(j+2)

  idcg = 0
  for j,p in enumerate(ideal):
    if(p['is_selected']==1):
      idcg += 1/np.log2(j+2)

  if(idcg > 0):
    total += 1
    score += dcg/idcg

print('NDCG = ', score/total)


pred_correct = 0
correct = 0
for row in data:
  pred = sorted(row['passages'], key=lambda x:x['score'], reverse=True)
  ideal = sorted(row['passages'], key=lambda x:x['is_selected'], reverse=True)
  for i in range(5):
    if(pred[i]['is_selected']==1):
      pred_correct += 1
    if(ideal[i]['is_selected']==1):
      correct += 1
print('recall@5 = ',pred_correct/correct)


pred_correct = 0
correct = 0
for row in data:
  pred = sorted(row['passages'], key=lambda x:x['score'], reverse=True)
  ideal = sorted(row['passages'], key=lambda x:x['is_selected'], reverse=True)
  if(pred[0]['is_selected']==1):
    pred_correct += 1
  if(ideal[0]['is_selected']==1):
    correct += 1

print('recall@1 = ',pred_correct/correct)
print('precision@1 = ',pred_correct/len(data))
