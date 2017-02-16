#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <Sift.h>

typedef map<string, vector<SiftDescriptor> > Vector_Descriptor;

class bow : public Classifier
{
public:
  bow(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  Vector_Descriptor descriptor_set;
  
  virtual void train(const Dataset &filenames) 
  { 
    ofstream myfile,csvfile;
	
	myfile.open("bow_svm.txt");
	csvfile.open("bow_csv.csv");
	
	vector<SiftDescriptor> K_vector;
	
	// Put all the siftDescriptor in one vector for k-means 
	

    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
		cout << "Processing " << c_iter->first << endl;
	
		for(int i=0; i<c_iter->second.size(); i++){
			CImg<double> input_image = extract_features(c_iter->second[i].c_str());
		    
			vector<SiftDescriptor> input_descriptors = Sift::compute_sift(input_image);
			
			stringstream ind;
	  		ind << i;
	  		string str = ind.str();
			string key = c_iter->first +"_" +str;
			descriptor_set[key] = input_descriptors;
			
			for(int de =0; de < input_descriptors.size(); ++de){
				for(int k=0; k< 128; ++k){
					csvfile << input_descriptors[de].descriptor[k] << ";";
					}
				csvfile << "\n";
				}
			}

		}
		csvfile.close();
		system("python3 k_means.py");

		
	  

	// k-means implementation


	// Creating Histogram of visual vocabulary



	for(Vector_Descriptor::const_iterator iter = descriptor_set.begin(); iter != descriptor_set.end(); ++iter)
      {	
      	CImg<double> bow_vectors(k_size,1,1);
      	vector<SiftDescriptor> test_descriptor = iter->second;

      	ofstream test_csv;
      	test_csv.open("test_csv.csv", ofstream::out | ofstream::trunc);

      	for(int de =0; de < test_descriptors.size(); ++de){
				for(int k=0; k< 128; ++k){
					test_csv << test_descriptors[de].descriptor[k] << ";";
					}
				test_csv << "\n";
				}
		test_csv.close();

		system("python3 k_means_predict.py")
	  	ifstream infile("image_centroid.txt");
	  	
	  	string str;
	  	int index;
	  	for(int count = 0; count < test_descriptors.size(); ++count){
	  		str << infile;
	  		std::istringstream iss(str);
	  		iss >> index;
	  		bow_vectors(index,0) = bow_vectors.atXY(index,0)+1;
	  	}
	  	infile.close();

	  	int ind = c_iter->first.find("_");
	  	string class_str = c_iter->first.substr(0,ind);
	  	

	  	myfile<<std::find(class_list.begin(), class_list.end(), class_str) - class_list.begin() + 1;
	 	myfile<<' ';
		for(int j=0; j<k_size; j++){
			myfile<<j+1<<':'<<bow_vectors(j,0);
			myfile<<' ';
			}
	    myfile<<'\n';
	  }	
	
	myfile.close();
	system("./svm_multiclass_learn -c 1.0 bow_svm.txt svm_struct_model_bow");
}
	  
	 
	CImg<double> RGB_to_Gray(CImg<double> image) {

		CImg<double> gray(image.width(), image.height(), 1, 1, 0);

		cimg_forXY(image, x, y) {

			gray(x, y, 0, 0) = 0.33*image(x,y,0,0) + 0.33*image(x,y,0,1) + 0.33*image(x,y,0,2);
		}
		//gray.save("gray.png");
		return gray;
	}





  virtual string classify(const string &filename,const string &clas,const Dataset &filenames)
  {
    ofstream myfile_test;
	
	myfile_test.open("test.txt");
	CImg<double> test_image = extract_features(filename);
	// CImg<double> test_gray = test_image.get_RGBtoHSI().get_channel(2);
	vector<SiftDescriptor> test_descriptors = Sift::compute_sift(test_image);

	CImg<double> test_bow(k_size,1,1);


	ofstream test_csv;
     test_csv.open("test_csv.csv", ofstream::out | ofstream::trunc);

     for(int de =0; de < test_descriptors.size(); ++de){
		for(int k=0; k< 128; ++k){
			test_csv << test_descriptors[de].descriptor[k] << ";";
			}
		test_csv << "\n";
		}
	test_csv.close();

	system("python3 k_means_predict.py")
  	
  	ifstream infile("image_centroid.txt");
  	
  	string str;
  	int index;
  	for(int count = 0; count < test_descriptors.size(); ++count){
  		str << infile;
  		std::istringstream iss(str);
  		iss >> index;
  		test_bow(index,0) = test_bow.atXY(index,0)+1;
  	}
  	infile.close();
	  	
	  


	// prepare bag of words for test image
	
	
	  
    for(int j=0; j<k_size; j++){
		myfile_test<<j+1<<':'<<test_bow(j,0);
		myfile_test<<' ';
	}
		
	  myfile_test<<'\n';
	  myfile_test.close();
	  system("./svm_multiclass_classify test.txt svm_struct_model_bow");
	
	fstream myfile_pred("svm_predictions", std::ios_base::in);
	int pred_value;
	myfile_pred>>pred_value;
	myfile_pred.close();
	
	return class_list[pred_value-1];
   
  }

  virtual void load_model()
  {
    /*for(int c=0; c < class_list.size(); c++)
      models[class_list[c] ] = (CImg<double>(("nn_model." + class_list[c] + ".png").c_str()));*/
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
      // return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
      CImg<double> test(filename.c_str());
	   CImg<double> temp = RGB_to_Gray(test);
		// CImg<double> mean("mean.png");
		//CImg<double> centered = subtract_mean(temp.resize(size, size, 1, 1).unroll('x'), mean);
		//return centered;
		temp.resize(size, size, 1, 1);
		return temp;
    }

  static const int size=40;  // subsampled image resolution
  static const int k_size = 100;
  map<string, CImg<double> > models; // trained models
  map<string,string> target_value;
  map<string,string> itarget_value;
};
