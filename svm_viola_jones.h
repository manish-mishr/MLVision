#include <fstream>
#include <iostream>
#include <string>
class svm_viola_jones : public Classifier
{
public:
  svm_viola_jones(const vector<string> &_class_list) : Classifier(_class_list) {}
  
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
	 
	  system("./svm_multiclass_learn -c 0.1 -e 0.5  svmlight.txt");
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
  CImg<double> get_grayScale(CImg<double> ip_image)
  {
    CImg<double> t(ip_image.width(),ip_image.height());
				 
	for(int i=0;i<ip_image.width();i++){
		for(int j=0;j<ip_image.height();j++){
			t(i,j)=(0.299*ip_image(i,j,0,0))+(0.587*ip_image(i,j,0,1))+(0.114*ip_image(i,j,0,2));	
		}
	}
	return t;
  }
  
  CImg<double> znorm(CImg<double> gscale)
  {
	CImg<double> znormed(gscale.width(),gscale.height());
	double var = gscale.variance();
	double mn  = gscale.mean();
	double sd = sqrt(var);
	//cout<<"mean"<<mn<<endl;
	//cout<<"sd"<<sd<<endl;
	for(int i=0;i<gscale.width();i++){
		for(int j=0;j<gscale.height();j++){
			znormed(i,j) = (gscale(i,j) - mn)/sd;
			//cout<<"znormed"<<znormed(i,j)<<endl;
		}
	}	
	return znormed;
  }
  
  CImg<double> get_integral_image(CImg<double> gscale)
  {
	CImg<double> integral(gscale.width(),gscale.height());
	
	for(int i=0;i<gscale.height();i++){
		for(int j=0;j<gscale.width();j++){
			double temp_integral=gscale(j,i);
			if((i-1) >= 0){
				temp_integral = temp_integral + integral(j,i-1); 
			}
			if((j-1) >= 0){
				temp_integral = temp_integral + integral(j-1,i);
			}
			if(((i-1) >= 0) && ((j-1) >= 0) ){
				temp_integral = temp_integral - integral(j-1,i-1);
			}
			//cout<<temp_integral;
			integral(j,i)=temp_integral;
		}
	}
	return integral;
  }
  
  CImg<double> get_haar_features(string pattern,CImg<double> integral_image,int nrow,int ncol)
  {
	srand (time(NULL));
	int xr=5 + ( std::rand() % ( 15 -5 + 1 ) );
	if((xr%2) != 0){
	xr++;
	}
	srand (time(NULL));
	int xc=5 + ( std::rand() % ( 15 -5 + 1 ) );
	if((xc%2) != 0){
	xc++;
	}
	CImg<double> pat(6,6);
	int offsetRow = 0;
	int offsetCol = 0;
	vector<char> v;
	
	for ( std::string::iterator it=pattern.begin(); it!=pattern.end(); ++it)
		v.push_back(*it);

	 for(int i=0;i<v.size();i++){
		for(int j=0;j<pat.height()/nrow;j++){
			for(int z=0;z<pat.width()/ncol;z++){
				if(v[i] == 'a'){
				pat(z+offsetCol,j+offsetRow)=1;
				}
				else{
				pat(z+offsetCol,j+offsetRow)=-1;
				}
			}
			
		}
		offsetCol = offsetCol + pat.width()/ncol;
		//offsetRow = offsetRow + pat.height()/nrow;
		
		if(offsetCol >= pat.width()){
			offsetRow = offsetRow + pat.height()/nrow;
			offsetCol = 0;
		}
		
	}
	
	//pat.save("pat.jpg");
	return pat;
	
  }
  
  CImg<double> get_features(vector<CImg<double> > pattern,CImg<double> integral_image)
  {
	CImg<double> features((pattern.size()*size)*size,1);
	//CImg<double> features(150,1);
	int feat_count=0;
	
	for(int z=0;z<pattern.size();z++){
	for(int x=0;x<integral_image.height();x++){
		for(int y=0;y<integral_image.width();y++){
			features(feat_count,0) = 0;
			for(int i=0;i<pattern[z].height();i++){
				for(int j=0;j<pattern[z].width();j++){
					if ((x + i) < integral_image.height() && (y + j) < integral_image.width()){ 
					features(feat_count,0) = features(feat_count,0) + (integral_image(y+j,x+i) * pattern[z](j,i));}
					
				}
				
			}
			
			feat_count++;
		}
	}
	}
	
	return features;
		
  }
  
  CImg<double> extract_features(const string &filename)
    {
	  CImg<double> ip_image(filename.c_str());
	  ip_image=ip_image.resize(size,size);
	  CImg<double> gscale=get_grayScale(ip_image);
	  //gscale=gscale.normalize(-1,1);
	  gscale=znorm(gscale);
	  
	  
	  CImg<double> integral_image=get_integral_image(gscale);
	  CImg<double> pattern1 = get_haar_features("abba",integral_image,2,2);
	  CImg<double> pattern2 = get_haar_features("aba",integral_image,3,1);
	  CImg<double> pattern3 = get_haar_features("ab",integral_image,2,1);
	  CImg<double> pattern4 = get_haar_features("aba",integral_image,1,3);
	  CImg<double> pattern5 = get_haar_features("ab",integral_image,1,2);
	  CImg<double> pattern6 = get_haar_features("aabb",integral_image,2,2);
	  CImg<double> pattern7 = get_haar_features("bbaa",integral_image,2,2);
	  CImg<double> pattern8 = get_haar_features("abab",integral_image,2,2);
	  CImg<double> pattern9 = get_haar_features("baab",integral_image,2,2);
	  CImg<double> pattern10 = get_haar_features("bab",integral_image,1,3);
	  CImg<double> pattern11 = get_haar_features("bab",integral_image,3,1);
	  CImg<double> pattern12 = get_haar_features("aab",integral_image,1,3);
	  CImg<double> pattern13 = get_haar_features("aab",integral_image,3,1);
	  CImg<double> pattern14 = get_haar_features("ba",integral_image,1,2);
	  CImg<double> pattern15 = get_haar_features("ba",integral_image,2,1);
	  vector<CImg<double> > pattern;
	  pattern.push_back(pattern1);
	  pattern.push_back(pattern2);
	  pattern.push_back(pattern3);
	  pattern.push_back(pattern4);
	  pattern.push_back(pattern5);
	  pattern.push_back(pattern6);
	  pattern.push_back(pattern7);
	  pattern.push_back(pattern8);
	  pattern.push_back(pattern9);
	  pattern.push_back(pattern10);
	  pattern.push_back(pattern11);
	  pattern.push_back(pattern12);
	  pattern.push_back(pattern13);
	  pattern.push_back(pattern14);
	  pattern.push_back(pattern15);
	  CImg<double> features = get_features(pattern,integral_image);
	  
	  //features = features.normalize(0,1);
	  //features.save("harr.jpg");
	  return features.unroll('x');
	  //gscale.save_png("integral.png");
	  //get_summed_area_table(CImg<double>(filename.c_str()).resize(size,size,1,3));
      //return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
    }
  
  
  static const int size=60;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
  map<string,string> target_value;
  map<string,string> itarget_value;
};
