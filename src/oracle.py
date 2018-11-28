import numpy
import foolbox
import scipy.stats as stats

# hard coded at the moment
# change .argsort()[128]

def least_confidence(predictions):
    
    #get the most confident class
    best_class = numpy.amax(predictions, axis=1)

    # and then get the 128 least confident images
    image_indices = best_class.argsort()[:128]
    
    return image_indices

def margin_sampling(predictions):
    
    # first let's grab the two best classes
    sorted_array = numpy.sort(predictions, axis=1)
    best_class = sorted_array[:, -2:]

    # now let's substract them
    # since they are sorted, we are sure that [:, 1] > [:, 0], no negative values should be there
    best_class = numpy.subtract(best_class[:, 1], best_class[:, 0])
    
    # and then the 128 images with the smallest margin
    image_indices = best_class.argsort()[:128]

    return image_indices

def entropy(predictions):

    # first let's calculate entropy for each images
    pred_entropy = []
    for position in range(predictions.shape[0]):
        pred_entropy.append(stats.entropy(predictions[position]))
    
    pred_entropy = numpy.array(pred_entropy)
    
    # and now we take the greatest entropy
    image_indices = pred_entropy.argsort()[-128:]

    return image_indices

def adversarial_margin(model, image_array, label_array):
    raise NotImplementedError('sorry this method is not implemented correctly')
    # instanciate keras model & define attack
    foolbox_model = model = foolbox.models.KerasModel(model, bounds=(0, 1))
    attack = foolbox.attacks.FGSM(foolbox_model)
    # attack each images and save the distance
    image_distance = []

    for position in range(image_array.shape[0]):
        adversarial = attack(image_array[position], numpy.argmax(label_array[position]), unpack=False)
        # WTF, this return a class/function ???
        distance = adversarial.distance # Mean squared distance (L²)
        #if(distance )
        image_distance.append(distance)

    image_distance = numpy.array(image_distance)
    # sorting images: we select the 128 images with the smallest distance
    image_indices = image_distance.argsort()[:128]

    return image_indices