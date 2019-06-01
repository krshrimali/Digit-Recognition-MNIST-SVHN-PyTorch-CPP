/* Author: Kushashwa, http://www.github.com/krshrimali
 * Reference: https://pytorch.org/cppdocs/frontend.html
 */

#include <torch/torch.h>

/* Sample code for training a FCN on MNIST dataset using PyTorch C++ API */

struct Net: torch::nn::Module {
	Net() {
		// Register 3 FC modules
		fc1 = register_module("fc1", torch::nn::Linear(784, 64));
		fc2 = register_module("fc2", torch::nn::Linear(64, 32));
		fc3 = register_module("fc3", torch::nn::Linear(32, 10));
	}

	// Implement algorithm
	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
		x = torch::dropout(x, 0.5, is_training());
		x = torch::relu(fc2->forward(x));
		x = torch::log_softmax(fc3->forward(x), 1);
		return x;
	}

	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main() {
	auto net = std::make_shared<Net>();

	// Create multi-threaded data loader for MNIST data
	auto data_loader = torch::data::make_data_loader(
			torch::data::datasets::MNIST("../data").map(
				torch::data::transforms::Stack<>()),64);
	torch::optim::SGD optimizer(net->parameters(), 0.01);

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

