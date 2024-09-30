SHOW TABLES;

SELECT * FROM TB_USERLEVEL;
DROP TABLE TB_USERLEVEL;
CREATE TABLE TB_USERLEVEL (
		USERLEVEL_CODE		VARCHAR(2)		NOT NULL
	,	USERLEVEL_NAME		VARCHAR(20)		NOT NULL
);

ALTER TABLE TB_USERLEVEL ADD PRIMARY KEY (USERLEVEL_CODE);

INSERT INTO TB_USERLEVEL VALUES('00','영구정지회원');
INSERT INTO TB_USERLEVEL VALUES('01','정지회원');
INSERT INTO TB_USERLEVEL VALUES('02','일반회원');
INSERT INTO TB_USERLEVEL VALUES('03','정회원');
INSERT INTO TB_USERLEVEL VALUES('04','프리미엄회원');
INSERT INTO TB_USERLEVEL VALUES('05','유저관리자');
INSERT INTO TB_USERLEVEL VALUES('06','영업관리자');
INSERT INTO TB_USERLEVEL VALUES('07','개발자');
INSERT INTO TB_USERLEVEL VALUES('08','마스터관리자');

COMMIT;

SELECT * FROM TB_USERREGTYPE;
DROP TABLE TB_USERREGTYPE;
CREATE TABLE TB_USERREGTYPE (
		USERREG_CODE		VARCHAR(2)		NOT NULL
	,	USERREG_NAME		VARCHAR(20)		NOT NULL
);

ALTER TABLE TB_USERREGTYPE ADD PRIMARY KEY (USERREG_CODE);

INSERT INTO TB_USERREGTYPE VALUES('00','구글');
INSERT INTO TB_USERREGTYPE VALUES('01','네이버');
INSERT INTO TB_USERREGTYPE VALUES('02','카카오');

COMMIT;

SELECT * FROM TB_USERCONFIRM;
DROP TABLE TB_USERCONFIRM;
CREATE TABLE TB_USERCONFIRM (
		USERCONFIRM_CODE	VARCHAR(2)		NOT NULL
	,	USERCONFIRM_CHECK	VARCHAR(10)		NOT NULL
);

ALTER TABLE TB_USERCONFIRM ADD PRIMARY KEY (USERCONFIRM_CODE);

INSERT INTO TB_USERCONFIRM VALUES('00','미승인');
INSERT INTO TB_USERCONFIRM VALUES('01','승인');

COMMIT;

SELECT * FROM TB_USERS;
DROP TABLE TB_USERS;
CREATE TABLE TB_USERS (
		USER_CODE			VARCHAR(2)		NOT NULL
	,	USERLEVEL_CODE		VARCHAR(20)		NOT NULL
	,	USERREG_CODE		VARCHAR(20)		NOT NULL
	,	USERCONFIRM_CODE	VARCHAR(2)		NOT NULL
	,   USER_ID				VARCHAR(30)		NOT NULL
	,	USER_NICKNAME		VARCHAR(50)		NULL
	,	USER_PHONE			VARCHAR(30)		NULL
	,	USER_EMAIL			VARCHAR(30)		NULL
	,	USER_EPD			DATE			NULL
);

ALTER TABLE TB_USERS ADD PRIMARY KEY (USER_CODE);
ALTER TABLE TB_USERS ADD CONSTRAINT TB_USER_USERLEVEL_CODE_FK FOREIGN KEY (USERLEVEL_CODE) REFERENCES TB_USERLEVEL(USERLEVEL_CODE);
ALTER TABLE TB_USERS ADD CONSTRAINT TB_USER_USERREG_CODE_FK FOREIGN KEY (USERREG_CODE) REFERENCES TB_USERREGTYPE(USERREG_CODE);
ALTER TABLE TB_USERS ADD CONSTRAINT TB_USER_USERCONFIRM_CODE_FK FOREIGN KEY (USERCONFIRM_CODE) REFERENCES TB_USERCONFIRM(USERCONFIRM_CODE);

COMMIT;

SELECT * FROM TB_USERACCESSHISTORY;
DROP TABLE TB_USERACCESSHISTORY;
CREATE TABLE TB_USERACCESSHISTORY (
		USER_CODE			VARCHAR(2)		NOT NULL
	,	USERLEVEL_CODE		VARCHAR(20)		NOT NULL
	,	UESRACCESS_DATE		DATE			NOT NULL	DEFAULT		SYSDATE()
);

ALTER TABLE TB_USERACCESSHISTORY ADD CONSTRAINT TB_USERACCESSHISTORY_USER_CODE_FK FOREIGN KEY (USER_CODE) REFERENCES TB_USERS(USER_CODE);
ALTER TABLE TB_USERACCESSHISTORY ADD CONSTRAINT TB_USERACCESSHISTORY_USERLEVEL_CODE_FK FOREIGN KEY (USERLEVEL_CODE) REFERENCES TB_USERS(USERLEVEL_CODE);

COMMIT;

SELECT * FROM TB_USERCONFIRMHISTORY;
DROP TABLE TB_USERCONFIRMHISTORY;
CREATE TABLE TB_USERCONFIRMHISTORY (
		USER_CODE				VARCHAR(2)		NOT NULL
	,	USERCONFIRM_CODE		VARCHAR(20)		NOT NULL
	,	USERCONFIRM_DATE		DATE			NOT NULL	DEFAULT		SYSDATE()
);

ALTER TABLE TB_USERCONFIRMHISTORY ADD CONSTRAINT TB_USERCONFIRMHISTORY_USER_CODE_FK FOREIGN KEY (USER_CODE) REFERENCES TB_USERS(USER_CODE);
ALTER TABLE TB_USERCONFIRMHISTORY ADD CONSTRAINT TB_USERCONFIRMHISTORY_USERLEVEL_CODE_FK FOREIGN KEY (USERCONFIRM_CODE) REFERENCES TB_USERS(USERCONFIRM_CODE);

COMMIT;

SELECT * FROM TB_INVESTCLASSIFICATION;
DROP TABLE TB_INVESTCLASSIFICATION;
CREATE TABLE TB_INVESTCLASSIFICATION(
		INVEST_CODE			VARCHAR(2)		NOT NULL
	,	INVEST_NAME			VARCHAR(20)		NOT NULL
);

ALTER TABLE TB_INVESTCLASSIFICATION ADD PRIMARY KEY (INVEST_CODE);

INSERT INTO TB_INVESTCLASSIFICATION VALUES('01','주식');
INSERT INTO TB_INVESTCLASSIFICATION VALUES('02','상품');
INSERT INTO TB_INVESTCLASSIFICATION VALUES('03','지수');
INSERT INTO TB_INVESTCLASSIFICATION VALUES('04','지표');

COMMIT;

SELECT * FROM TB_STOCKCLASSIFY;
DROP TABLE TB_STOCKCLASSIFY;
CREATE TABLE TB_STOCKCLASSIFY (
		GSTC_CODE		VARCHAR(20)		NOT NULL
	,	INVEST_CODE		VARCHAR(2)		NOT NULL
	,	KSTC_CODE		VARCHAR(20)		NOT NULL
	,	STC_NAME		VARCHAR(20)		NOT NULL
);

ALTER TABLE TB_STOCKCLASSIFY ADD PRIMARY KEY (GSTC_CODE);
ALTER TABLE TB_STOCKCLASSIFY ADD CONSTRAINT TB_STOCKCLASSIFY_INVEST_CODE_FK FOREIGN KEY (INVEST_CODE) REFERENCES TB_INVESTCLASSIFICATION(INVEST_CODE);

COMMIT;

SELECT * FROM TB_PRODUCTCLASSIFY;
DROP TABLE TB_PRODUCTCLASSIFY;
CREATE TABLE TB_PRODUCTCLASSIFY(
		PDT_CODE		VARCHAR(20)		NOT NULL
	,	INVEST_CODE		VARCHAR(2)		NOT NULL
	,	PDT_NAME		VARCHAR(20)		NOT NULL
	,	PDT_TICKER		VARCHAR(20)		NOT NULL
);

ALTER TABLE TB_PRODUCTCLASSIFY ADD PRIMARY KEY (PDT_CODE);
ALTER TABLE TB_PRODUCTCLASSIFY ADD CONSTRAINT TB_PRODUCTCLASSIFY_INVEST_CODE_FK FOREIGN KEY (INVEST_CODE) REFERENCES TB_INVESTCLASSIFICATION(INVEST_CODE);

COMMIT;

SELECT * FROM TB_INDEXCLASSIFY;
DROP TABLE TB_INDEXCLASSIFY;
CREATE TABLE TB_INDEXCLASSIFY(
		GINDEX_CODE		VARCHAR(20)		NOT NULL
	,	INVEST_CODE		VARCHAR(2)		NOT NULL
	,	GINDEX_NAME		VARCHAR(20)		NOT NULL
	,	GINDEX_TICKER	VARCHAR(20)		NOT NULL
);

ALTER TABLE TB_INDEXCLASSIFY ADD PRIMARY KEY (GINDEX_CODE);
ALTER TABLE TB_INDEXCLASSIFY ADD CONSTRAINT TB_INDEXCLASSIFY_INVEST_CODE_FK FOREIGN KEY (INVEST_CODE) REFERENCES TB_INVESTCLASSIFICATION(INVEST_CODE);

COMMIT;

##################################################################################################### 작성 중
SELECT * FROM TB_INDICATOR;
DROP TABLE TB_INDICATOR;
CREATE TABLE TB_INDICATOR (
		USAINDICATOR_CODE	VARCHAR(20)		NOT NULL
	,	INVEST_CODE			VARCHAR(2)
	,	USINDICATOR_DATE	DATE			NOT NULL
	,	USINDICATOR_VALUE	VARCHAR(30)				
);

ALTER TABLE TB_INDICATOR ADD CONSTRAINT TB_INDICATOR_USINDICATOR_CODE_FK FOREIGN KEY (USINDICATOR_CODE) REFERENCES TB_INDEXCLASSIFY(USINDICATOR_CODE);
ALTER TABLE TB_INDICATOR ADD CONSTRAINT TB_INDICATOR_INVEST_CODE_FK FOREIGN KEY (INVEST_CODE) REFERENCES TB_INVESTCLASSIFICATION(INVEST_CODE);

SELECT * FROM TB_DAILYINDEX
DROP TABLE TB_DAILYINDEX
CREATE TABLE TB_DAILYINDEX(

);

SELECT * FROM TB_DAILYPRODUCT;
DROP TABLE TB_DAILYPRODUCT;
CREATE TABLE TB_DAILYPRODUCT(

);

SELECT * FROM TB_DAILYSTOCK;
DROP TABLE TB_DAILYSTOCK;
CREATE TABLE TB_DAILYSTOCK(

);

SELECT * FROM TB_MINSSTOCK;
DROP TABLE TB_MINSSTOCK;
CREATE TABLE TB_MINSSTOCK(

);

SELECT * FROM TB_KRNEWS;
DROP TABLE TB_KRNEWS;
CREATE TABLE TB_KRNEWS(

);

SELECT * FROM TB_USNEWS;
DROP TABLE TB_USNEWS;
CREATE TABLE TB_USNEWS(

);

