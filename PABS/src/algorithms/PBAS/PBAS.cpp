#include "PBAS.h"

using namespace bgslibrary::algorithms::pbas;

//这里有个初始化列表，Segmenter.cpp里面也有一个初始化列表
PBAS::PBAS() : 
  N(20), R_lower(18), Raute_min(2), 
  T_lower(2), T_upper(200), R_scale(5), 
  R_incdec(0.05), T_dec(0.05), T_inc(1)
{
  //feature vector
  alpha = 7.0;
  beta = 1.0;
  formerMeanNorm = 0;
  width = 0;

  //result image
  foregroundValue = 255;
  backgroundValue = 0;

  //length of random array
  //哪个array有这么长的长度？
  countOfRandomNumb = 1000;

  //the T(x_i) value needs initiation
  //T_init用distance threshold的下界进行初始化的
  T_init = R_lower;

  //check if something is moving in the picture
  //这个isMove是通过什么方式判断的，可能要关注一下
  isMove = false;

  //for init, count number of runs
  runs = 0;
  newInitialization();
}

//这些vector的长度都为 countOfRandomNumb,并且都按照均一分布，插入了值
void PBAS::newInitialization()
{
  if (!randomN.empty())
    randomN.clear();

  if (!randomX.empty())
    randomX.clear();

  if (!randomY.empty())
    randomY.clear();

  if (!randomMinDist.empty())
    randomMinDist.clear();

  if (!randomT.empty())
    randomT.clear();

  if (!randomTN.empty())
    randomTN.clear();

  for (int l = 0; l < countOfRandomNumb; l++)
  {
    randomN.push_back((int)randomGenerator.uniform((int)0, (int)N));
    //这个没根据图像实际的长宽去限制，是用checkvaild来确保的
    randomX.push_back((int)randomGenerator.uniform(-1, +2));
    randomY.push_back((int)randomGenerator.uniform(-1, +2));
    randomMinDist.push_back((int)randomGenerator.uniform((int)0, (int)N));
    //randomT长度是countOfRandomNumb,里面是0-T_upper间的随机数
    randomT.push_back((int)randomGenerator.uniform((int)0, (int)T_upper));
    randomTN.push_back((int)randomGenerator.uniform((int)0, (int)T_upper));
  }
}

PBAS::~PBAS(void)
{
  randomN.clear();
  randomX.clear();
  randomY.clear();
  randomMinDist.clear();
  randomT.clear();
  randomTN.clear();

  for (int k = 0; k < backgroundModel.size(); ++k)
  {
    if (chans == 1)
    {
      backgroundModel.at(k).at(0).release();
      backgroundModel.at(k).at(1).release();
    }
    else
    {
      backgroundModel.at(k).at(0).release();
      backgroundModel.at(k).at(1).release();
      backgroundModel.at(k).at(2).release();

      backgroundModel.at(k).at(3).release();
      backgroundModel.at(k).at(4).release();
      backgroundModel.at(k).at(5).release();
    }
  }

  backgroundModel.clear();
  meanMinDist.release();

  actualR.release();
  actualT.release();

  sobelX.release();
  sobelY.release();
}

//知道他说的只有background才会更新是什么意思了
//通过if (count >= Raute_min)来控制
//如果判断为前景，就不会进入这个if语句，相应的就不会执行更新的操作
bool PBAS::process(cv::Mat* input, cv::Mat* output)
{
  if (width != input->cols)
  {
    width = input->cols;
    chans = input->channels();
    height = input->rows;

    if (input->rows < 1 || input->cols < 1)
    {
      std::cout << "Error: Occurrence of to small (or empty?) image size in PBAS. STOPPING " << std::endl;
      return false;
    }
  }

  //iniate the background model
  //如果第一次运行，就把所有值设置为默认值
  //其他情况下基本就是计算sobelX,sobelY的模值
  //细节在这个里面，如果runs = N了以后
  //这个init就不在++runs了
  init(input);

  resultMap = new cv::Mat(input->rows, input->cols, CV_8UC1);

  //calculate features
  //这个也是在算梯度的模值
  calculateFeatures(&currentFeatures, input);

  //set sumMagnitude to zero at beginning and then sum up in the loop
  sumMagnitude = 0;
  long glCounterFore = 0;
  isMove = false;

  //Here starts the whole processing of each pixel of the image
  // for each pixel
  for (int j = 0; j < resultMap->rows; ++j)
  {
    //这个是指向每行的开头
    resultMap_Pt = resultMap->ptr<uchar>(j);
    currentFeaturesM_Pt.clear();
    currentFeaturesC_Pt.clear();
    std::vector<float*> fT;
    std::vector<uchar*> uT;
    B_Mag_Pts.clear();
    B_Col_Pts.clear();

    for (int z = 0; z < chans; ++z)
    {
      //currentFeaturesM_Pt里面装的是梯度值
      //currentFeaturesC里面是每个channel的灰度值
      currentFeaturesM_Pt.push_back(currentFeatures.at(z).ptr<float>(j));
      currentFeaturesC_Pt.push_back(currentFeatures.at(z + chans).ptr<uchar>(j));

      B_Mag_Pts.push_back(fT);

      B_Col_Pts.push_back(uT);
    }

    meanMinDist_Pt = meanMinDist.ptr<float>(j);
    actualR_Pt = actualR.ptr<float>(j);
    actualT_Pt = actualT.ptr<float>(j);

    //这个N array的background model其实里面也就是每帧的feature
    for (int k = 0; k < runs; ++k)
    {
      for (int z = 0; z < chans; ++z)
      {
        //backgroundModel里面一样最开始的是梯度的模值，后面依次是每个通道的灰度值
        B_Mag_Pts.at(z).push_back(backgroundModel.at(k).at(z).ptr<float>(j));
        B_Col_Pts.at(z).push_back(backgroundModel.at(k).at(z + chans).ptr<uchar>(j));
      }
    }

    //此for循环往下就是每一个像素位置了
    for (int i = 0; i < resultMap->cols; ++i)
    {
      //Compare each pixel to in the worst runtime-case each background model
      int count = 0;
      int index = 0;

      double norm = 0.0;
      double dist = 0.0;
      double minDist = 1000.0;
      int entry = randomGenerator.uniform(3, countOfRandomNumb - 4);

      //这个do循环维护的是表示background dynamic的dmin
      //这个for循环每次改变的是index，就是之前的每一帧和当前帧进行比较
      do
      {
        if (chans == 3)
        {
          norm = sqrt(
            (((double)B_Mag_Pts.at(0).at(index)[i] - ((double)*currentFeaturesM_Pt.at(0)))*((double)B_Mag_Pts.at(0).at(index)[i] - ((double)*currentFeaturesM_Pt.at(0)))) +
            (((double)B_Mag_Pts.at(1).at(index)[i] - ((double)*currentFeaturesM_Pt.at(1)))*((double)B_Mag_Pts.at(1).at(index)[i] - ((double)*currentFeaturesM_Pt.at(1)))) +
            (((double)B_Mag_Pts.at(2).at(index)[i] - ((double)*currentFeaturesM_Pt.at(2)))*((double)B_Mag_Pts.at(2).at(index)[i] - ((double)*currentFeaturesM_Pt.at(2))))
          );

          dist = sqrt(
            (((double)B_Col_Pts.at(0).at(index)[i] - ((double)*currentFeaturesC_Pt.at(0)))*((double)B_Col_Pts.at(0).at(index)[i] - ((double)*currentFeaturesC_Pt.at(0)))) +
            (((double)B_Col_Pts.at(1).at(index)[i] - ((double)*currentFeaturesC_Pt.at(1)))*((double)B_Col_Pts.at(1).at(index)[i] - ((double)*currentFeaturesC_Pt.at(1)))) +
            (((double)B_Col_Pts.at(2).at(index)[i] - ((double)*currentFeaturesC_Pt.at(2)))*((double)B_Col_Pts.at(2).at(index)[i] - ((double)*currentFeaturesC_Pt.at(2))))
          );
        }
        else
        {
          //一个是梯度模值上的距离，还有个是灰度值间的距离
          //被减的就是当前帧对应的梯度模值和图像灰度值
          //index从下面来看是和runs挂钩的
          norm = abs((((double)B_Mag_Pts.at(0).at(index)[i] -
            ((double)*currentFeaturesM_Pt.at(0)))*((double)B_Mag_Pts.at(0).at(index)[i] - ((double)*currentFeaturesM_Pt.at(0)))));

          dist = abs((((double)B_Col_Pts.at(0).at(index)[i] -
            ((double)*currentFeaturesC_Pt.at(0)))*((double)B_Col_Pts.at(0).at(index)[i] - ((double)*currentFeaturesC_Pt.at(0))))
          );
        }
        // 3.5 Implement details里面的表达式
        //论文里面是一句话，也没细说
        //他这里的norm前的系数，formerMeanMag是N帧里面所有的没有落在距离阈值范围内的像素算的
        dist = ((double)alpha*(norm / formerMeanMag) + beta*dist);

        //在（i,j)这个像素位置，如果距离小于设定的阈值，count就会记录
        if ((dist < *actualR_Pt))
        {
          ++count;
          //minDist总是记录下最小的dist
          //所以这个minDist始终就是只有一个值
          //确实应该是这样，论文里面说的维护一个minDist的array明显不太靠谱
		  if (minDist > dist)
            minDist = dist;
        }
        else
        {
          //glCounterForce是在记录超过距离阈值的次数
          //只有在dist大于distance threshold的地方才会记录sumMagnitude,
          //这个最终用来更新计算dist时候用到的formerMeanMag
          sumMagnitude += (double)(norm);
          ++glCounterFore;
        }
        ++index;
      } while ((count < Raute_min) && (index < runs));
      //do while退出的条件也值得分析以下，如果count大于这个#mim(Raute_min)了，那就可以直接退出了

      //#############################################
      //update backgroundmodel
      // is BACKGROUND
      //#min这个东西是这样用的吗，比如说我当前跑了这么多次
      //如果distance小于distance threshold的次数大于两次，就可以认为是背景了
      if (count >= Raute_min)
      {
        *resultMap_Pt = 0;
        // 这个ratio在这里起到什么作用？
        // actualT_pt的值在后面的regulator里面更新
        double ratio = std::ceil((double)T_upper / (double)(*actualT_Pt));
        //in the first run every distance is zero, because there is no background model
        //in the secont run, we have already one image as background model, hence a
        // reasonable minDist could be found -> because of the partly 1/run changing in the running average, we set in the first try meanMinDist to the actual minDist value
        if (runs < N && runs > 2)
        {
          //meanMinDist是这样维护的
          *meanMinDist_Pt = ((((float)(runs - 1)) * (*meanMinDist_Pt)) + (float)minDist) / ((float)runs);
        }
        else if (runs < N && runs == 2)
        {
          //第二次没球mean，直接就是actual的minDist
          *meanMinDist_Pt = (float)minDist;
        }

        //1. update model
        //这里只是单纯的把N array里面的数值给进行更新了
        if (runs == N)
        {
          //Update current pixel
          //check if random numer is smaller than ratio
          //randomT是（0，200）间的值
          //这个if语句的意义，我理解的就是实现一个概率之类的东西
          //不是每次运行到这里都会执行
          if (randomT.at(entry) < ratio)
          {
            // replace randomly chosen sample
            int rand = randomN.at(entry + 1); //randomGenerator.uniform((int)0,(int)N-1);
            for (int z = 0; z < chans; ++z)
            {
              //at(z)没什么好说的，这个是和channel对应的
              //at(rand)比较重要，是从N中取一个k，替换为新的像素值
              //[i]取行指针指向的具体的像素值
              B_Mag_Pts.at(z).at(rand)[i] = (float)*currentFeaturesM_Pt.at(z);
              B_Col_Pts.at(z).at(rand)[i] = (uchar)*currentFeaturesC_Pt.at(z);

            }

            *meanMinDist_Pt = ((((float)(N - 1)) * (*meanMinDist_Pt)) + (float)minDist) / ((float)N);
          }

          //Update neighboring pixel model
          if (randomTN.at(entry) < ratio)
          {
            //choose neighboring pixel randomly
            int xNeigh = randomX.at(entry) + i;
            int yNeigh = randomY.at(entry) + j;
            checkValid(&xNeigh, &yNeigh);

            // replace randomly chosen sample
            int rand = randomN.at(entry - 1);
            for (int z = 0; z < chans; ++z)
            {
              //at(rand)是取了N中第rand次的currentFeature
              //at(z)是currentFeature中对应的channel
              //at(yNeigh,xNeigh)取出对应的像素
              (backgroundModel.at(rand)).at(z).at<float>(yNeigh, xNeigh) = currentFeatures.at(z).at<float>(yNeigh, xNeigh);
              (backgroundModel.at(rand)).at(z + chans).at<uchar>(yNeigh, xNeigh) = currentFeatures.at(z + chans).at<uchar>(yNeigh, xNeigh);
            }
          }
        }
      }
      else
      {
        // 这个是说N array里面，没有一个和当前像素的距离是小于设定的距离阈值的
        // 或者说只有1个是小于距离阈值的，其他都比距离阈值大
        // store pixel as foreground
        *resultMap_Pt = 255;

        //there is some movement
        isMove = true;
      }

      //#######################//#######################//#######################//#######################
      //control loops
      //#######################//#######################//#######################//#######################
      //update R
      decisionThresholdRegulator(actualR_Pt, meanMinDist_Pt);

      //update T
      learningRateRegulator(actualT_Pt, meanMinDist_Pt, resultMap_Pt);

      //#######################//#######################//#######################//#######################
      //#######################//#######################//#######################//#######################

      //jump to next pixel
      ++resultMap_Pt;
      for (int z = 0; z < chans; ++z)
      {
        ++currentFeaturesM_Pt.at(z);
        ++currentFeaturesC_Pt.at(z);
      }

      ++meanMinDist_Pt;
      ++actualR_Pt;
      ++actualT_Pt;
    }
  }

  //resultMap里面应该不是0就是255
  resultMap->copyTo(*output);

  //if there is no foreground -> no magnitudes fount
  //-> initiate some low value to prevent diving through zero
  double meanMag = sumMagnitude / (double)(glCounterFore + 1); //height*width);

  if (meanMag > 20)
    formerMeanMag = meanMag;
  else
    formerMeanMag = 20;

  delete resultMap;

  for (int z = 0; z < chans; ++z)
  {
    currentFeatures.at(z + chans).release();
    currentFeatures.at(z).release();
  }

  return true;
}

void PBAS::decisionThresholdRegulator(float* pt, float* meanDist)
{
  //update R
  double tempR = *pt;
  double newThresh = (*meanDist)*R_scale;

  if (tempR < newThresh)
  {
    tempR += tempR * R_incdec;
  }
  else
  {
    tempR -= tempR * R_incdec;
  }

  if (tempR >= R_lower)
    *pt = (float)tempR;
  else
    *pt = (float)R_lower;
}

void PBAS::learningRateRegulator(float* pt, float* meanDist, uchar* isFore)
{
  //time update
  double tempT = *pt;

  if ((int)*isFore < 128)
  {
    tempT -= T_inc / (*meanDist + 1.0);
  }
  else
  {
    tempT += T_dec / (*meanDist + 1.0);
  }

  if (tempT > T_lower && tempT < T_upper)
    *pt = (float)tempT;
}

void PBAS::checkValid(int *x, int *y)
{
  if (*x < 0)
  {
    *x = 0;
  }
  else if (*x >= width)
  {
    *x = width - 1;
  }

  if (*y < 0)
  {
    *y = 0;
  }
  else if (*y >= height)
  {
    *y = height - 1;
  }
}

//计算一下sobelX，sobelY的模值
//然后放到backgroundModel这个vector里面
void PBAS::init(cv::Mat* input)
{
  if (runs < N)
  {
    std::vector<cv::Mat> init;
    //init里面装的就是sobelX和sobelY的模值
    calculateFeatures(&init, input);
    backgroundModel.push_back(init);

    if (chans == 1)
    {
      init.at(0).release();
      init.at(1).release();
    }
    else
    {
      init.at(0).release();
      init.at(1).release();
      init.at(2).release();
      init.at(3).release();
      init.at(4).release();
      init.at(5).release();
    }

    init.clear();

    //也就是run=0的时候会执行这个操作
    if (runs == 0)
    {
      meanMinDist.create(input->size(), CV_32FC1);
      meanMinDist.zeros(input->rows, input->cols, CV_32FC1);

      actualR.create(input->rows, input->cols, CV_32FC1);
      actualT.create(input->rows, input->cols, CV_32FC1);

      float* ptRs, *ptTs; //, *ptM;
      //一开始的R和T全部填的是默认值
      for (int rows = 0; rows < actualR.rows; ++rows)
      {
        //行指针
        ptRs = actualR.ptr<float>(rows);
        ptTs = actualT.ptr<float>(rows);

        for (int cols = 0; cols < actualR.cols; ++cols)
        {
          ptRs[cols] = (float)R_lower;
          ptTs[cols] = (float)T_init;
        }
      }
    }

    ++runs;
  }
}

//利用sobel算子去计算梯度，实际有梯度的模值和方向
// 三通道的就分成3张图，分别去进行计算
// 方向信息dir怎么好像没用到，只把对应的模值给放进mag里面了
// 先放的梯度模值，后面依次是三个通道的灰度值
void PBAS::calculateFeatures(std::vector<cv::Mat>* feature, cv::Mat* inputImage)
{
  if (!feature->empty())
    feature->clear();

  cv::Mat mag[3], dir;

  if (inputImage->channels() == 3)
  {
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(*inputImage, rgbChannels);

    for (int l = 0; l < 3; ++l)
    {
      cv::Sobel(rgbChannels.at(l), sobelX, CV_32F, 1, 0, 3, 1, 0.0);
      cv::Sobel(rgbChannels.at(l), sobelY, CV_32F, 0, 1, 3, 1, 0.0);

      // Compute the L2 norm and direction of the gradient
      cv::cartToPolar(sobelX, sobelY, mag[l], dir, true);
      feature->push_back(mag[l]);
      sobelX.release();
      sobelY.release();
    }

    feature->push_back(rgbChannels.at(0));
    feature->push_back(rgbChannels.at(1));
    feature->push_back(rgbChannels.at(2));
    rgbChannels.at(0).release();
    rgbChannels.at(1).release();
    rgbChannels.at(2).release();
  }
  else
  {
    cv::Sobel(*inputImage, sobelX, CV_32F, 1, 0, 3, 1, 0.0);
    cv::Sobel(*inputImage, sobelY, CV_32F, 0, 1, 3, 1, 0.0);

    // Compute the L2 norm and direction of the gradient
    cv::cartToPolar(sobelX, sobelY, mag[0], dir, true);
    feature->push_back(mag[0]);

    cv::Mat temp;
    inputImage->copyTo(temp);
    feature->push_back(temp);
    temp.release();
  }

  mag[0].release();
  mag[1].release();
  mag[2].release();
  dir.release();
}

void PBAS::setN(int temp)
{
  N = temp;
  newInitialization();
}

void PBAS::setRaute_min(int temp)
{
  Raute_min = temp;
}

void PBAS::setR_lower(double temp)
{
  R_lower = temp;
}

void PBAS::setR_incdec(double temp)
{
  R_incdec = temp;
}

void PBAS::setR_scale(double temp)
{
  R_scale = temp;
}

void PBAS::setT_init(double temp)
{
  T_init = temp;
}

void PBAS::setT_lower(double temp)
{
  T_lower = temp;
}

void PBAS::setT_upper(double temp)
{
  T_upper = temp;
  newInitialization();
}

void PBAS::setT_dec(double temp)
{
  T_dec = temp;
}

void PBAS::setT_inc(double temp)
{
  T_inc = temp;
}

void PBAS::setAlpha(double temp)
{
  alpha = temp;
}

void PBAS::setBeta(double temp)
{
  beta = temp;
}

bool PBAS::isMovement()
{
  return isMove;
}

//cv::Mat* PBAS::getR1_xi()
//{
//	return &actualR;
//}
//
//cv::Mat* PBAS::getT_xi()
//{
//	return &actualT;
//}
