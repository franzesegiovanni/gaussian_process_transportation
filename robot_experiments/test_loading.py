import pickle

with open("results/dressing/target_0.pkl","rb") as source:
                source_distribution = pickle.load(source)


print(source_distribution)

