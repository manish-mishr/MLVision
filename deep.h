#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
class deep : public Classifier
{
public:
  deep(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // Nearest neighbor training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames) 
  { 
    ofstream myfile,outfile;
	
	myfile.open("overfeat.txt");

    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
		cout << "Processing " << c_iter->first << endl;
		CImg<double> class_vectors(size, size, 1);
		
	//CImg<double> class_vectors(size*size*3, filenames.size(), 1);
	
	// convert each image to be a row of this "model" image

		for(int i=0; i<c_iter->second.size(); i++){
			CImg<double> input_image(c_iter->second[i].c_str());
			input_image.save("of_image.png");
	  		class_vectors = class_vectors.draw_image(0, 0, 0, 0, extract_features(c_iter->second[i].c_str()));
	  		cout  << "image:  " << c_iter->second[i] << endl;
	  		string name = "Over_model" + c_iter->first;
	  		stringstream ind;
	  		ind << i;
	  		string str = ind.str();
	  		name = name + str;
			class_vectors.save_png(( name + ".png").c_str());
			name = name + ".png";
			outfile.open("overfeat_feature.txt", ofstream::out | ofstream::trunc);
	  		system(("./overfeat/bin/linux_64/overfeat -L 18 of_image.png  >> overfeat_feature.txt").c_str());
	  		outfile.close();
	  		ifstream infile("overfeat_feature.txt");
	  		string line;
	  		int dimension,row,col;
	  		getline(infile,line);
	  		std::istringstream iss(line);
	  		iss >> dimension >> row >> col;
	  		CImg<double> second_vectors(dimension,1,1);
	  		int index = 0;
	  		while(!infile.eof()){
	  			infile >> second_vectors(index,0);
	  			++index;
	  		}
	  		
	  		myfile<<std::find(class_list.begin(), class_list.end(), c_iter->first) - class_list.begin() + 1;
	  		myfile<<' ';
	  		for(int j=0; j<dimension; ++j){
				myfile<<j+1<<':'<< second_vectors;
				myfile<<' ';
			}
	 		myfile<<'\n';
	  
		}
		
      }
  		myfile.close();
  		system("./svm_multiclass_learn -c 1.0 overfeat.txt ");
	} 
	  


  virtual string classify(const string &filename,const string &clas,const Dataset &filenames)
  {
    ofstream myfile_test,myfile_over;
    myfile_test.open("test.txt");
	myfile_over.open("test_over.txt");
	
	CImg<double> test_vectors(size, size, 1);
	CImg<double> test_image(size,size,1);
	test_vectors = test_vectors.draw_image(0, 0, 0, 0, extract_features(filename));
	test_vectors.save_png("test_image.png");
	
	system("./overfeat/bin/linux_64/overfeat -L 18 test_image.png  >> test_over.txt");
	myfile_test<<std::find(class_list.begin(), class_list.end(), clas) - class_list.begin() +1;
	  myfile_test<<' ';

	ifstream infile("test_over.txt");
	string line;
	int dimension,row,col;
	getline(infile,line);
	std::istringstream iss(line);
	iss >> dimension >> row >> col;
	// CImg<double> second_vectors(dimension,1,1);
	// double temp;
	// int index = 0;
	// while(infile >> temp){
	// 	second_vectors(index,0) = temp ;
	// 	++index;
	// }
	double temp;
	for(int j=0; j<dimension; j++){
		infile >> temp;
		myfile_test<<j+1<<':'<< temp;
		myfile_test<<' ';
		}
		
	  myfile_test<<'\n';
	  myfile_test.close();
	  system("./svm_multiclass_classify test.txt svm_struct_model");
	
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
	   temp.resize(size, size, 1, 1);
		return temp;
    }

  static const int size=231;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
  map<string,string> target_value;
  map<string,string> itarget_value;
};
