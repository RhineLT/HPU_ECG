import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from help_code_demo_tf import ECG_DataSET, ToTensor, create_dataset
from models.model_tf import AFNet, AFNet_Com, AFNet_light

# Define TeacherNet and StudentNet
from models.model_tf import TeacherNet, StudentNet

def distillation_loss(y_true, y_pred_student, y_pred_teacher, temperature=3, alpha=0.5):
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    # Standard cross entropy loss
    student_loss = loss_fn(y_true, y_pred_student)
    
    # KL divergence loss
    y_pred_teacher_soft = tf.nn.softmax(y_pred_teacher / temperature)
    y_pred_student_soft = tf.nn.softmax(y_pred_student / temperature)
    kl_loss = losses.KLDivergence()(y_pred_teacher_soft, y_pred_student_soft) * (temperature ** 2)
    
    return alpha * student_loss + (1 - alpha) * kl_loss

def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []
    Test_fb = []
    best_fb = 0.0
    best_acc = 0.0
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiate Teacher and Student networks
    teacher_net = TeacherNet()
    teacher_net.build((1, 1250, 1, 1))
    # teacher_net.load_weights('./ECG_student_net_tf.h5')  # Load pretrained weights
    teacher_net.trainable = False  # Freeze teacher network

    student_net = StudentNet()
    student_net.build((1, 1250, 1, 1))
    print(student_net.summary())

    optimizer = optimizers.Adam(learning_rate=LR)
    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)

    # Start dataset loading
    trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE, transform=ToTensor())
    trainloader = create_dataset(trainset, BATCH_SIZE)

    testset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE, transform=ToTensor())
    testloader = create_dataset(testset, BATCH_SIZE)

    print("Start training")
    for epoch in range(EPOCH):
        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for step, (x, y) in enumerate(trainloader):
            with tf.GradientTape() as tape:
                logits_student = student_net(x, training=True)
                logits_teacher = teacher_net(x, training=False)
                loss = distillation_loss(y, logits_student, logits_teacher)
                grads = tape.gradient(loss, student_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, student_net.trainable_variables))
                pred = tf.argmax(logits_student, axis=1)
                correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
                accuracy += correct / x.shape[0]
                correct = 0.0

                running_loss += loss
                i += 1
        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))

        Train_loss.append(running_loss / i)
        Train_acc.append(accuracy / i)

        y_true = []
        y_pred = []
        running_loss = 0.0
        accuracy = 0.0
        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        for x, y in testloader:
            logits_student = student_net(x, training=False)
            test_loss = loss_object(y, logits_student)
            pred = tf.argmax(logits_student, axis=1)
            total += y.shape[0]
            correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
            running_loss_test += test_loss
            i += x.shape[0]
            y_true.extend(y.numpy())
            y_pred.extend(pred.numpy())

        test_accuracy = correct / total
        test_loss_avg = running_loss_test / i
        print('Test Acc: %.5f Test Loss: %.5f' % (test_accuracy, test_loss_avg))
        Test_loss.append(test_loss_avg)
        Test_acc.append(test_accuracy)
        fb = fbeta_score(y_true, y_pred, average='weighted', beta=2)
        Test_fb.append(fb)
        print('F-beta score: %.5f' % fb)
        # Save the model for each epoch
        student_net.save(f'./saved_models/{epoch+1}_ECG_student_net_tf.h5')

        # Save the model if the test accuracy is the best we've seen so far
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            student_net.save('./saved_models/best_ECG_student_net_tf.h5')
        if fb > best_fb:
            best_fb = fb
            student_net.save('./saved_models/fb_ECG_student_net_tf.h5')
    # Save final model and results
    student_net.save('./saved_models/ECG_student_net_tf.h5')

    # Write results to file
    with open('./saved_models/loss_acc.txt', 'w') as file:
        file.write("Train_loss\n")
        file.write(str(Train_loss))
        file.write('\n\n')
        file.write("Train_acc\n")
        file.write(str(Train_acc))
        file.write('\n\n')
        file.write("Test_loss\n")
        file.write(str(Test_loss))
        file.write('\n\n')
        file.write("Test_acc\n")
        file.write(str(Test_acc))
        file.write('\n\n')

    print('Finish training')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=16)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='/tf/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    args = argparser.parse_args()

    main()
