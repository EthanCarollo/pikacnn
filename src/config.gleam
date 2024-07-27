pub const data_path = "./data/pokemon-augmented-clean-resized"

// The image_size is also the size of the input model
pub const image_size = 150

// This value is here to say how much images you want to get loaded for every label, in our case
// for a little training, we will just put 50 to not saturate our ram, but in fact we can put a very 
// higher number without problem if we have the necessary memory
pub const max_image_per_label = 20

pub const epoch = 50

pub const batch_size = 32

pub const learning_rate = 0.001
