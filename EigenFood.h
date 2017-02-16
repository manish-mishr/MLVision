#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

class EigenFood: public Classifier {

public:
	EigenFood(const vector<string> &_class_list) :
		Classifier(_class_list) {
	}

	void print_CImg(CImg<double> &img) {

		for (int y = 0; y < img._height; y++) {
			for (int x = 0; x < img._width; x++) {
				cout << img(x, y, 0, 0) << "\t";
			}
			cout << endl;
		}
	}

	vector<string> files_in_dir(const string &directory,
			bool prepend_directory = false) {
		vector<string> file_list;
		DIR *dir = opendir(directory.c_str());
		if (!dir)
			throw std::string("Can't find directory " + directory);

		struct dirent *dirent;
		while ((dirent = readdir(dir)))
			cout << "Image " << endl;
		if (dirent->d_name[0] != '.')
			file_list.push_back(
					(prepend_directory ? (directory + "/") : "")
							+ dirent->d_name);

		closedir(dir);
		return file_list;
	}

	CImg<double> RGB_to_Gray(CImg<double> image) {

		CImg<double> gray(image.width(), image.height(), 1, 1, 0);

		cimg_forXY(image, x, y) {

			gray(x, y, 0, 0) = 0.33*image(x,y,0,0) + 0.33*image(x,y,0,1) + 0.33*image(x,y,0,2);
		}
		//gray.save("gray.png");
		return gray;
	}

	CImgList<double> preprocess_for_eigenfood(const Dataset &filenames) {

		CImgList<double> images;
		for (Dataset::const_iterator c_iter = filenames.begin(); c_iter
				!= filenames.end(); ++c_iter) {
			int count = 0;
			cout << "Processing " << c_iter->first << endl;

			for (int i = 0; i < c_iter->second.size(); i++) {

				CImg<double> input_image(c_iter->second[i].c_str());
				CImg<double> gray_image = RGB_to_Gray(input_image);
				gray_image = gray_image.resize(size, size, 1, 1);
				CImg<double> class_vector = gray_image.unroll('x');
				class_vector.transpose();
				images.push_back(class_vector);
				//				if (count > 1)
				//				break;
				count++;
			}
		}
		
		return images;
	}

	void calculate_mean(CImg<double> &matrix, CImg<double> &mean) {

		cout << "calculating average" << endl;
		//cout << "matrix " << matrix._height << ":" << matrix._width << endl;
		for (int y = 0; y < matrix._height; y++) {
			double sum = 0;
			for (int x = 0; x < matrix._width; x++) {
				//cout << "x:y ::" << x << ":" << y << endl;
				sum += matrix(x, y, 0, 0);
			}
			double avg = sum / matrix.width();
			mean(0, y, 0, 0) = avg;
		}
	}

	CImg<double> subtract_mean(CImg<double> &matrix, CImg<double> &mean) {

		CImg<double> centered_matrix(matrix.width(), matrix.height(), 1, 1, 0);
		cout << "subtracting" << endl;
		for (int y = 0; y < matrix._height; y++) {
			for (int x = 0; x < matrix._width; x++) {
				centered_matrix(x, y, 0, 0) = matrix(x, y, 0, 0) - mean(0, y,
						0, 0);
			}
		}
		return centered_matrix;
	}

	CImg<double> multiply(CImg<double> &A, CImg<double> &B) {
		CImg<double> covariance(B._width, A._height, 1, 1, 0);
		for (int i = 0; i < A._height; i++) {
			for (int j = 0; j < B._width; j++) {
				for (int k = 0; k < B._height; k++) {
					covariance(j, i, 0, 0) += A(k, i, 0, 0) * B(j, k, 0, 0);
				}
			}
		}
		return covariance;
	}

	CImgList<double> roll(CImg<double> image) {

		CImgList<double> list;
		for (int x = 0; x < image.width(); x++) {
			CImg<double> new_image(size, size, 1, 1);
			int j = 0;
			int k = 0;
			for (; j < image.height();) {
				for (int i = 0; i < size; i++) {
					new_image(i, k, 0, 0) = image(x, j, 0, 0);
					j++;
				}
				k++;
			}
			list.push_back(new_image.normalize(0,255));
		}
		return list;
	}

	void add(CImg<double> image) {
		for (int i = 0; i < image.width(); i++) {
			double sum = 0;
			for (int j = 0; j < image.height(); j++) {
				sum += image(j, i, 0, 0);
			}
			cout << "sum:" << sum << endl;
		}
	}

	void save_eigen_vectors(CImgList<double> vector_images) {

		CImg<double> temp = vector_images.get_append('x');
		int i = 0;
        cimglist_for(vector_images, img) {
		    CImg<double> image = vector_images[img];
	    	string x = "eigenVectors/vector";
	    	std::stringstream out;
	    	out << i;
		    string y = out.str();
	    	string z = ".png";

		    image.save((x + y + z).c_str());
	    	i++;
	    }
    }

	void 
	save_in_file(CImg<double> eigenvectors, string filename) {
		
		ofstream myfile;
		myfile.open(filename.c_str());
		myfile << eigenvectors.height() << " " << eigenvectors.width() << endl;
		for(int i = 0; i < eigenvectors.height(); i++) {
			for(int j = 0; j < eigenvectors.width(); j++) {
				myfile << eigenvectors(j, i, 0, 0) << " ";
			}
			myfile << endl;
		}
	}
	
	CImg<double> split(CImg<double> eigen_vectors, int k) {
		
		CImg<double> new_image(k, eigen_vectors.height(), 1, 1);
		for(int i = 0; i < k; i++)
		{
			for(int j = 0; j < eigen_vectors.height(); j++)
			{
				new_image(i, j, 0, 0) = eigen_vectors(i, j, 0, 0);
			}
		}
		return new_image;
	}
	
	CImg<double> eigenFood(const Dataset &filenames) {

		CImgList<double> imageList = preprocess_for_eigenfood(filenames);
		CImg<double> images = imageList.get_append('x');

		//Calculating mean
		CImg<double> mean(1, images.height(), 1, 1, 0);
		calculate_mean(images, mean);
		mean.save("mean.png");
		CImgList<double> list = roll(mean);
		list[0].save("mean_food.png");
		//print_CImg(mean);

		CImg<double> matrix = subtract_mean(images, mean);
		cout << "matrix :: " << matrix._height << ":"<< matrix._width << endl;
		//print_CImg(matrix);

		CImg<double> matrix_T = matrix.get_transpose();
		cout << "matrix_t :" << matrix_T._height << ":" << matrix_T._width << endl;

		//CImg<double> covariance = multiply(matrix_T, matrix);
		CImg<double> covariance = multiply(matrix, matrix_T);
		cout << "covariance :" << covariance._height << ":" << covariance._width << endl;

		CImg<double> val;
		CImg<double> vec;
		covariance.symmetric_eigen(val, vec);
		save_in_file(val, "eigen_values.txt");
		cout << "vec :" << vec._height << ":" << vec._width << endl;
		cout << "vectors" << endl;
		//print_CImg(vec);
		//add(vec);
		CImg<double> eigen_vectors = vec;
		//CImg<double> eigen_vectors = multiply(matrix, vec);
		cout << "eigen_vectors :: " << eigen_vectors._height << ":" << eigen_vectors._width
				<< endl;
		
		CImg<double> k_eigen_vectors = split(eigen_vectors, k);
		save_in_file(k_eigen_vectors, "eigenvectors.txt");
		
		cout << "k_eigen_vectors :: " << k_eigen_vectors._height << ":" << k_eigen_vectors._width
			<< endl;
		//cout << "k_eigen vectors" << endl;
		//print_CImg(k_eigen_vectors[0]);
		CImgList<double> vector_images = roll(k_eigen_vectors);
		save_eigen_vectors(vector_images);

		CImg<double> eigenfoods = vector_images.get_append('x');
		eigenfoods.save("eigenfood.png");

		//CImg<double> weights = multiply(k_eigen_vectors[0].transpose(), matrix);
		CImg<double> weights = multiply(k_eigen_vectors.transpose(), images);
		cout << "weigths :: " << weights._height << ":" << weights._width
			<< endl;

		weights.transpose();
		CImgList<double> w = weights.get_split('x', weights.width());
		cout << "weigths" << w.size() << endl;
		
		cout << "weigths :: " << weights._height << ":" << weights._width
			<< endl;
		cout << "weigths " << endl;
	//	print_CImg(weights);

		return weights;
	}

	virtual void train(const Dataset &filenames) {

		CImg<double> weights = eigenFood(filenames);

		//weights.normalize(-100, 100);
		ofstream myfile;
		myfile.open("eigenfoods_train.txt");

		int y = 0;
		for (Dataset::const_iterator c_iter = filenames.begin(); c_iter
				!= filenames.end(); ++c_iter) {
			for (int i = 0; i < c_iter->second.size(); i++) {

				myfile << std::find(class_list.begin(), class_list.end(),
						c_iter->first) - class_list.begin() + 1;
				myfile << ' ';

				for(int w = 0; w < weights.width(); w++) {
					myfile << w + 1 << ':' << weights(w, y, 0, 0);
					myfile << ' ';
				}
				y++;
				myfile << '\n';
			}
		}
		myfile.close();
		system("./svm_multiclass_learn -c 0.1 eigenfoods_train.txt eigenfoods_model");
	}

	CImg<double> extract_features(const string &filename) {
		CImg<double> test(filename.c_str());
		CImg<double> temp = RGB_to_Gray(test);
		// CImg<double> mean("mean.png");
		//CImg<double> centered = subtract_mean(temp.resize(size, size, 1, 1).unroll('x'), mean);
		//return centered;
		temp.resize(size, size, 1, 1);
		return temp.unroll('x').transpose();
	}

	virtual string classify(const string &filename, const string &clas,
			const Dataset &filenames) {
		ofstream myfile_test;

		myfile_test.open("test.txt");
		CImg<double> image = extract_features(filename);
		//cout << "image :: " << image._height << ":" << image._width
		//	<< endl;

		CImg<double> test_image = multiply(eigenVectors, image);
		test_image.transpose();
//		test_image.normalize(-100, 100);
		//cout << "test_image :: " << test_image._height << ":" << test_image._width
		//	<< endl;
		//print_CImg(eigenVectors);
		myfile_test << std::find(class_list.begin(), class_list.end(), clas)
		- class_list.begin() + 1;
		myfile_test << ' ';

		for (int j = 0; j < test_image.width(); j++) {
			//cout << test_image(j, 0) << " ";
			myfile_test << j + 1 << ':' << test_image(j, 0);
			myfile_test << ' ';
		}
		cout << endl;

		myfile_test << '\n';
		myfile_test.close();
		system("./svm_multiclass_classify test.txt eigenfoods_model");

		fstream myfile_pred("svm_predictions", std::ios_base::in);
		int pred_value;
		myfile_pred >> pred_value;
		myfile_pred.close();

		return class_list[pred_value - 1];
	}
	
	virtual void load_model() {
		
		int height, width;
		ifstream myReadFile("eigenvectors.txt");
		myReadFile >> height >> width;
		CImg<double> temp(width, height, 1, 1);
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				double x;
				myReadFile >> x;
				temp(j, i) = x;
			}
		}
		eigenVectors = temp;
		eigenVectors.transpose();
		//cout << "eigenVectors :: " << eigenVectors._height << ":" << eigenVectors._width
		//	<< endl;
	
	}

	protected:

		static const int k = 1250;
		static const int size = 40;
		CImg<double> eigenVectors;
};

