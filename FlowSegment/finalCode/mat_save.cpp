#include "mat_head.h"

namespace yu
{
	std::string save_load_dirname = "F:\\MatlabDir\\";

	//depth=0:6依次对应CV_8U,CV_8S,CV_16U,CV_16S,CV_32S,CV_32F,CV_64F
	int  depthFromType(MatType type)
	{
		switch(type) {
		case UINT8: return 0; //CV_8U
		case INT8: return 1; //CV_8S
		case UINT16: return 2; //CV_16U
		case INT16: return 3; //CV_16S
		case INT32: return 4; //CV_32S
		case FLOAT32: return 5; //CV_32F
		case DOUBLE64: return 6; //CV_64F
		default:
			yuErr("depthFromType(): unsupported matrix type!");
		}
	}

	MatType  typeFromDepth(int depth)
	{
		switch(depth) {
		case 0: return UINT8;
		case 1: return INT8;
		case 2: return UINT16;
		case 3: return INT16;
		case 4: return INT32;
		case 5: return FLOAT32;
		case 6: return DOUBLE64;
		default:
			yuErr("typeFromDepth(): unsupported depth!");
		}
	}
	
	void  save(std::string name, const Mat &m)
	{
		//name = "G:\\PhD-data\\chapter3\\Figure\\FlowChart\\" + name;
		if(name.find(":") == std::string::npos)
			name = save_load_dirname + name;
		FILE *fid = fopen(name.c_str(), "wb");
		if(!fid)
			yuErr("save(): cannot open file \"%s\" to write data!", name.c_str());
		int depth = depthFromType(m.type);
		fwrite(&m.h, sizeof(int), 1, fid);
		fwrite(&m.w, sizeof(int), 1, fid);
		fwrite(&m.d, sizeof(int), 1, fid);
		fwrite(&depth, sizeof(int), 1, fid);
		if(m.isContinuous())
			fwrite(m.p, m.elemSz(), m.h * m.w * m.d, fid);
		else {
			uchar *p = m.p;
			for(int k = m.h; k--; p += m.step)
				fwrite(p, m.elemSz(), m.w * m.d, fid);
		}
		fclose(fid);
	}

	Mat  load(std::string name, bool throwError)
	{
		//name = "G:\\PhD-data\\chapter3\\Figure\\FlowChart\\" + name;
		if(name.find(":") == std::string::npos)
			name = save_load_dirname + name;
		FILE *fid = fopen(name.c_str(), "rb");
		if(!fid) {
			if(throwError)
				yuErr("load(): cannot open file \"%s\" to read data!", name.c_str());
			return Mat();
		}
		int rows(0), cols(0), channels(0), depth(0);
		fread(&rows, sizeof(int), 1, fid);
		fread(&cols, sizeof(int), 1, fid);
		fread(&channels, sizeof(int), 1, fid);
		fread(&depth, sizeof(int), 1, fid);
		MatType type = typeFromDepth(depth);
		Mat m(rows, cols, channels, type);
		size_t total_elements = size_t(rows) * size_t(cols) * size_t(channels);
		size_t elements_read = fread(m.p, m.elemSz(), total_elements, fid);
		if(elements_read != total_elements)
			yuErr("load(): elements_read != m.h * m.w * m.d)");
		fclose(fid);
		return m;
	}

	void  load(std::string name, bool throwError, yu::Mat &m)
	{
		if(name.find(":") == std::string::npos)
			name = save_load_dirname + name;
		FILE *fid = fopen(name.c_str(), "rb");
		if(!fid) {
			if(throwError)
				yuErr("load(): cannot open file \"%s\" to read data!", name.c_str());
			m.release();
			return;
		}
		int rows(0), cols(0), channels(0), depth(0);
		fread(&rows, sizeof(int), 1, fid);
		fread(&cols, sizeof(int), 1, fid);
		fread(&channels, sizeof(int), 1, fid);
		fread(&depth, sizeof(int), 1, fid);
		MatType type = typeFromDepth(depth);
		m.create(rows, cols, channels, type);
		size_t total_elements = size_t(rows) * size_t(cols) * size_t(channels);
		size_t elements_read = fread(m.p, m.elemSz(), total_elements, fid);
		if(elements_read != total_elements)
			yuErr("load(): elements_read != m.h * m.w * m.d)");
		fclose(fid);
	}

	//-------------- Methods to save/load very large matrix to/from local file. -----------

	void  save(FILE *fid, uchar *data, size_t elemSz, int64 nElems)
	{
		size_t batchSz = UINT32_MAX / elemSz; //num. elements to write each time
		int64 cnt = 0;
		while(cnt + (int64)batchSz < nElems) {
			size_t n = fwrite(data, elemSz, batchSz, fid);
			if(n != batchSz)
				yuErr("save(): n != batchSz");
			cnt += batchSz;
			data += batchSz * elemSz;
		}
		if(cnt < nElems) {
			size_t n = fwrite(data, elemSz, nElems - cnt, fid);
			if(n != nElems - cnt)
				yuErr("save(): n != nElems - cnt");
		}
	}

	void  load(FILE *fid, uchar *data, size_t elemSz, int64 nElems)
	{
		size_t batchSz = UINT32_MAX / elemSz; //num. elements to read each time
		int64 cnt = 0;
		while(cnt + (int64)batchSz < nElems) {
			size_t n = fread(data, elemSz, batchSz, fid);
			if(n != batchSz)
				yuErr("load(): n != batchSz");
			cnt += batchSz;
			data += batchSz * elemSz;
		}
		if(cnt < nElems) {
			size_t n = fread(data, elemSz, nElems - cnt, fid);
			if(n != nElems - cnt)
				yuErr("load(): n != nElems - cnt");
		}
	}

	void  saveL(std::string name, const LargeMat &m)
	{
		name = save_load_dirname + name;
		FILE *fid = fopen(name.c_str(), "wb");
		if(!fid)
			yuErr("saveL(): cannot open file \"%s\" to write data!", name.c_str());
		int h = (int)m.h; fwrite(&h, sizeof(int), 1, fid);
		int w = (int)m.w; fwrite(&w, sizeof(int), 1, fid);
		int d = (int)m.d; fwrite(&d, sizeof(int), 1, fid);
		int depth = depthFromType(m.type);
		fwrite(&depth, sizeof(int), 1, fid);
		size_t elemSz = (size_t)m.elemSz();
		if(m.isContinuous())
			save(fid, m.p, elemSz, m.h * m.w * m.d);
		else {
			uchar *p = m.p;
			for(int k = h; k--; p += m.step)
				save(fid, p, elemSz, m.w * m.d);
		}
		fclose(fid);
	}

	LargeMat  loadL(std::string name, bool throwError)
	{
		name = save_load_dirname + name;
		FILE *fid = fopen(name.c_str(), "rb");
		if(!fid) {
			if(throwError)
				yuErr("loadL(): cannot open file \"%s\" to read data!", name.c_str());
			return LargeMat();
		}
		int h; fread(&h, sizeof(int), 1, fid);
		int w; fread(&w, sizeof(int), 1, fid);
		int d; fread(&d, sizeof(int), 1, fid);
		int depth; fread(&depth, sizeof(int), 1, fid);
		MatType type = typeFromDepth(depth);
		LargeMat m((int64)h, (int64)w, (int64)d, type);
		size_t elemSz = (size_t)m.elemSz();
		if(m.isContinuous())
			load(fid, m.p, elemSz, m.h * m.w * m.d);
		else {
			uchar *p = m.p;
			for(int k = h; k--; p += m.step)
				load(fid, p, elemSz, m.w * m.d);
		}
		fclose(fid);
		return m;
	}
};

/*Matlab functions.

====================== 1 ======================
% function  yusave( name, A )
% % Save the Matlab matrix to local file, which could be loaded by yu::load().
% if ~ischar(name)
%     name_ = A;
%     A = name;
%     name = name_;
% end
% fid = fopen(name,'wb');
% [rows,cols,channels] = size(A);
% fwrite(fid,rows,'int');
% fwrite(fid,cols,'int');
% fwrite(fid,channels,'int');
% types = { 'uint8', 'int8', 'unint16', 'int16', 'int32', 'single', 'double' };
% precision = class(A);
% for k=1:length(types)
%     if strcmp(types{k},precision)==1
%         break;
%     end
% end
% depth = k - 1;
% fwrite(fid,depth,'int');
% B = zeros(cols*channels,rows);
% for k=1:channels
%     B(k:channels:end) = A(:,:,k)';
% end
% fwrite(fid,B(:),precision);
% fclose(fid);

====================== 2 ======================
% function  A = ld( name )
% % Load the matrix data from local file to Matlab storage. The data file is thought to be generated by yu::save().
% % clc;
% fp = fopen(name,'rb');
% rows = fread(fp,1,'int32');
% cols = fread(fp,1,'int32');
% channels = fread(fp,1,'int32');
% depth = fread(fp,1,'int32');%depth=0:6依次对应CV_8U,CV_8S,CV_16U,CV_16S,CV_32S,CV_32F,CV_64F
% if depth<0 || depth>6
%     error('Unknown data type!');
% end
% types = { 'uint8', 'int8', 'uint16', 'int16', 'int32', 'single', 'double' };
% precision = [types{depth+1} '=>' types{depth+1}];
% A = fread(fp,rows*cols*channels,precision); fclose(fp);
% X = reshape(A,[cols*channels,rows])';
% if channels==1
%     A = X;
%     return;
% end
% A = reshape(X,[rows,cols,channels]);
% for i=1:channels
%     A(:,:,i) = X(:,i:channels:end);
% end

====================== 3 ======================
% function  saveMatlab( name, A )
% % Unlike yusave(), this function will directly write matrix A into the file without rearranging the memory to accomodate for OpenCV style matrix.
% if ~ischar(name)
%     name_ = A;
%     A = name;
%     name = name_;
% end
% fid = fopen(name,'wb');
% [rows,cols,channels] = size(A);
% fwrite(fid,rows,'int');
% fwrite(fid,cols,'int');
% fwrite(fid,channels,'int');
% types = { 'uint8', 'int8', 'unint16', 'int16', 'int32', 'single', 'double' };
% precision = class(A);
% for k=1:length(types)
%     if strcmp(types{k},precision)==1
%         break;
%     end
% end
% depth = k - 1;
% fwrite(fid,depth,'int');
% fwrite(fid,A(:),precision);
% fclose(fid);

====================== 4 ======================
% function  A = loadMatlab( name )
% % 将saveMatlab函数保存的矩阵加载到matlab中
% % clc;
% name = ['C:\vot-toolkit-master\' name];
% fp = fopen(name,'rb');
% rows = fread(fp,1,'int32');
% cols = fread(fp,1,'int32');
% channels = fread(fp,1,'int32');
% depth = fread(fp,1,'int32');%depth=0:6依次对应CV_8U,CV_8S,CV_16U,CV_16S,CV_32S,CV_32F,CV_64F
% if depth<0 || depth>6
%     error('Unknown data type!');
% end
% types = { 'uint8', 'int8', 'uint16', 'int16', 'int32', 'single', 'double' };
% precision = [types{depth+1} '=>' types{depth+1}];
% sz = [rows,cols,channels];
% A = reshape(fread(fp,prod(sz),precision),sz);
% fclose(fp);

*/