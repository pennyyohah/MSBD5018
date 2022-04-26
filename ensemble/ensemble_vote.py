import json

with open("results/distilbert-spans-pred.txt") as distilbert,open("results/spanbert_token_spans-pred.txt") as spanbert,open("results/roberta_token_spans-pred.txt") as roberta, open("ensemble_vote.txt","w") as output:
    i = 0
    while True:
        distilbert_in = distilbert.readline()
        spanbert_in = spanbert.readline()
        roberta_in = roberta.readline()
        if not distilbert_in:
            break
        distilbert_in = set(json.loads(distilbert_in.split("\t")[1]))
        spanbert_in = set(json.loads(spanbert_in.split("\t")[1]))
        roberta_in = set(json.loads(roberta_in.split("\t")[1]))
        result1 = distilbert_in.intersection(spanbert_in)
        result2 = distilbert_in.intersection(roberta_in)
        result3 = spanbert_in.intersection(roberta_in)
        result = result1.union(result2).union(result3)
        result = list(result)
        result.sort()
        output.write(str(i)+"\t"+str(result)+"\n")
        i += 1      
