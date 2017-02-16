#include <fstream>
#include <iostream>
#include <string>
class svm : public Classifier
{
public:
  svm(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // Nearest neighbor training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames) 
  { 
    ofstream myfile;
	
	myfile.open("svmlight.txt");
	
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
		cout << "Processing " << c_iter->first << endl;
	
	
	//CImg<double> class_vectors(size*size*3, filenames.size(), 1);
	
	// convert each image to be a row of this "model" image
	for(int i=0; i<c_iter->second.size(); i++){
	  
	  CImg<double> class_vectors = extract_features(c_iter->second[i].c_str());
	  
	  myfile<<std::find(class_list.begin(), class_list.end(), c_iter->first) - class_list.begin() + 1;
	  myfile<<' ';
	  for(int j=0; j<class_vectors.width(); j++){
		myfile<<j+1<<':'<<class_vectors(j,0);
		myfile<<' ';
		}
	  myfile<<'\n';
	  }	
	
	
    }
	  myfile.close();
	 
	  system("./svm_multiclass_learn -c 1.0 svmlight.txt");
  }

  virtual string classify(const string &filename,const string &clas,const Dataset &filenames)
  {
    ofstream myfile_test;
	
	myfile_test.open("test.txt");
	CImg<double> test_image = extract_features(filename);
	
	myfile_test<<std::find(class_list.begin(), class_list.end(), clas) - class_list.begin() +1;
	  myfile_test<<' ';
	  
	  for(int j=0; j<test_image.width(); j++){
		myfile_test<<j+1<<':'<<test_image(j,0);
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
      return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
    }

  static const int size=20;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
  map<string,string> target_value;
  map<string,string> itarget_value;
};
