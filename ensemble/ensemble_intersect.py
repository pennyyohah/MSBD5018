import json

with open("results/distilbert-spans-pred.txt") as distilbert,open("results/spanbert_token_spans-pred.txt") as spanbert,open("results/roberta_token_spans-pred.txt") as roberta, open("ensemble_intersect.txt","w") as output:
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
        result = distilbert_in.intersection(spanbert_in)
        result = result.intersection(roberta_in)
        result = list(result)
        result.sort()
        output.write(str(i)+"\t"+str(result)+"\n")
        i += 1      
