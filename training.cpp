/* Author: Kushashwa, http://www.github.com/krshrimali
 * Reference: https://pytorch.org/cppdocs/frontend.html
 */

#include <torch/torch.h>

/* Sample code for training a FCN on MNIST dataset using PyTorch C++ API */

struct Net: torch::nn::Module {
    Net() {
        // Initialize CNN
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5)));
        conv2_drop = register_module("conv2_drop", torch::nn::Dropout());
        fc1 = register_module("fc1", torch::nn::Linear(320, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 10));
    }

    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x) {
        // std::cout << x.size(0) << ", " << 784 << std::endl;
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.view({-1, 320});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Dropout conv2_drop{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
	auto net = std::make_shared<Net>();

	// Create multi-threaded data loader for MNIST data
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(torch::data::datasets::MNIST("../data").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
				torch::data::transforms::Stack<>())), 64);
	torch::optim::SGD optimizer(net->parameters(), 0.01); // Learning Rate 0.01

	// net.train();

	for(size_t epoch=1; epoch<=10; ++epoch) {
		size_t batch_index = 0;
		// Iterate data loader to yield batches from the dataset
		for (auto& batch: *data_loader) {
			// Reset gradients
			optimizer.zero_grad();
			// Execute the model
			torch::Tensor prediction = net->forward(batch.data);
			// Compute loss value
			torch::Tensor loss = torch::nll_loss(prediction, batch.target);
			// Compute gradients
			loss.backward();
			// Update the parameters
			optimizer.step();

			// Output the loss and checkpoint every 100 batches
			if (++batch_index % 100 == 0) {
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index 
					<< " | Loss: " << loss.item<float>() << std::endl;
				torch::save(net, "net.pt");
			}
		}
	}
}

