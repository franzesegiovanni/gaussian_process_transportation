
import pickle
class BasicLoader:
    def __init__(self):
        super(BasicLoader, self).__init__()
    def load_distributions(self):
            try:
                with open("distributions/source.pkl","rb") as source:
                    self.source_distribution = pickle.load(source)
            except:
                print("No source distribution saved")

            try:
                with open("distributions/target.pkl","rb") as target:
                    self.target_distribution = pickle.load(target)
            except:
                print("No target distribution saved")    