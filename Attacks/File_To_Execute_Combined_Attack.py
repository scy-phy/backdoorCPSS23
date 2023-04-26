import numpy
import Combined_Backdoor_Attack
import Improved_Standard_Attack
import trainer_automatic

total = 0.0
best_accuracy = numpy.float64(1.0)  # best accuracy here means lowest accuracy

for i in range(0, 51):
    Combined_Backdoor_Attack.main(i)
    #Improved_Standard_Attack.main(i)
    accuracy, index = trainer_automatic.main(i)
    total += accuracy
    print("Accuracy: " + str(accuracy) + " with index: " + str(index))  # index tells me at which place the highest accuracy is and then I know which percentage was best
    print("Pattern " + str(i))
    if accuracy < best_accuracy:
        best_accuracy = accuracy

print("Best result is: " + str(best_accuracy))
print("Average: " + str(total / 51))
