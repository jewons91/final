select * from TB_MINSSTOCK tm ;
select * from TB_MINSSTOCK tm where tm.GSTC_CODE = 'KR7005930003' and tm.STCK_BSOP_DATE = '20240819';

SELECT MAX(a.STCK_CNTG_HOUR) AS STCK_CNTG_HOUR
FROM TB_MINSSTOCK a
WHERE a.GSTC_CODE = 'KR7005930003'
order by a.STCK_BSOP_DATE DESC ;

SELECT a.GSTC_CODE, a.STCK_BSOP_DATE , a.STCK_CNTG_HOUR , a.INSERTDATE 
FROM TB_MINSSTOCK a
where a.STCK_BSOP_DATE = '20240903'
order by a.INSERTDATE DESC ;

select *
  from TB_KRNEWS tk 
 order by KRNEWS_DATE DESC 
 limit 20;
 
SELECT *
  FROM TB_STOCKCLASSIFY ts ;
  
select *
  from TB_DAILYSTOCK td 
 where td.STCK_BSOP_DATE > '20200101' and gstc_code = 'KR7000660001'
 order by td.STCK_BSOP_DATE ASC;

select *
  from TB_MINSSTOCK tm
 where tm.STCK_BSOP_DATE = '20240919'
   and tm.GSTC_CODE = 'KR7003230000'
 order by STCK_CNTG_HOUR ASC ;

SELECT GSTC_CODE , count(*)
  from TB_MINSSTOCK tm 
 where STCK_BSOP_DATE = '20240904'
   and STCK_CNTG_HOUR = '130000'
 group by GSTC_CODE 
 having count(*)>1;

select count(*)
  from TB_MINSSTOCK tm 
<<<<<<< HEAD
 where STCK_BSOP_DATE = '20240919'
   and STCK_CNTG_HOUR = '150000';
=======
 where STCK_BSOP_DATE = '20240909';
--    and STCK_CNTG_HOUR = '150000';
>>>>>>> 257d464d1a7f4c3a6f1334ef70f0d9a5fb4f854f
 
select a.GSTC_CODE, a.STCK_PRPR, a.CNTG_VOL
  from TB_MINSSTOCK a
 where a.GSTC_CODE = 'KR7042700005'
   and a.stck_cntg_hour >= '090000'
   and a.stck_cntg_hour <= '110000'
 order by a.stck_bsop_date asc, a.stck_cntg_hour asc;
 
select a.GSTC_CODE, a.STCK_PRPR, a.CNTG_VOL, a.STCK_BSOP_DATE, a.STCK_CNTG_HOUR 
  from TB_MINSSTOCK a
 where a.GSTC_CODE = 'KR7000660001'
   and ((a.stck_cntg_hour >= '140000' && a.STCK_BSOP_DATE = '2024082') || (a.stck_cntg_hour <= '110000' && a.STCK_BSOP_DATE = '20240829')) 
 order by a.stck_bsop_date asc, a.stck_cntg_hour asc;
 
select a.GSTC_CODE, a.STCK_PRPR, a.CNTG_VOL, a.STCK_BSOP_DATE, a.STCK_CNTG_HOUR 
  from TB_MINSSTOCK a
 where a.GSTC_CODE = 'KR7000660001'
   and a.stck_cntg_hour >= '140000'
   and a.STCK_BSOP_DATE = '20240802'
UNION all
select a.GSTC_CODE, a.STCK_PRPR, a.CNTG_VOL, a.STCK_BSOP_DATE, a.STCK_CNTG_HOUR 
  from TB_MINSSTOCK a
 where a.GSTC_CODE = 'KR7000660001'
   and a.stck_cntg_hour <= '110000'
   and a.STCK_BSOP_DATE = '20240805'
   
 order by stck_bsop_date asc, stck_cntg_hour asc
;

CREATE table TB_STOCK_PREDICT(
	GSTC_CODE 				varchar(20) NOT NULL,
	INVEST_CODE 			varchar(2)	NOT NULL,
	PREDICT_RISE_RATE		varchar(30)	NOT NULL,
	PREDICT_NO_CHANGE_RATE	varchar(30)	NOT NULL,
	PREDICT_FALL_RATE		varchar(30)	NOT NULL,
	PREDICT_TIME			datetime	not null
);

select *
  from TB_STOCK_PREDICT;
  
select *
  from TB_MINSSTOCK tm 
 where GSTC_CODE = 'KR7005930003'
 order by tm.STCK_BSOP_DATE DESC, tm.STCK_CNTG_HOUR DESC
 limit 40;
 
SELECT td.STCK_BSOP_DATE, td.STCK_CNTG_HOUR, td.STCK_PRPR, td.STCK_LWPR, td.ACML_VOL
  FROM TB_MINSSTOCK td
 WHERE td.GSTC_CODE = 'KR7005930003'
 ORDER BY td.STCK_BSOP_DATE DESC, td.STCK_CNTG_HOUR DESC
 LIMIT 40;