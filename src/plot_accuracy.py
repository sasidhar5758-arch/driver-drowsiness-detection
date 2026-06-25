import matplotlib.pyplot as plt
import pickle

with open("history.pkl", "rb") as f:
    history = pickle.load(f)

plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

plt.savefig("outputs/accuracy.png")
plt.show()