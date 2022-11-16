library(rvest)
library(dplyr)


QUERY <- "서울보행환경" # 검색키워드
DATE  <- as.Date(as.character(20210727),format="%Y%m%d") # 검색시작날짜 & 검색종료날짜
DATE <- format(DATE, "%Y.%m.%d")
PAGE  <- 1

naver_url_1 <- "https://search.naver.com/search.naver?&where=news&query="
naver_url_2 <- "&pd=3&ds="
naver_url_3 <- "&de="
naver_url_4 <- "&start="

naver_url <- paste0(naver_url_1,QUERY,naver_url_2,DATE,naver_url_3,DATE,naver_url_4,PAGE)
naver_url


DATE_START <- as.Date(as.character(20210715),format="%Y%m%d") # 시작일자
DATE_END   <- as.Date(as.character(20210729),  format="%Y%m%d") # 종료일자
DATE <- DATE_START:DATE_END
DATE <- as.Date(DATE,origin="1970-01-01")
DATE


PAGE <- seq(from=1,to=41,by=10) # 시작값과 종료값을 지정
PAGE <- seq(from=1,by=10,length.out=5) # 시작값과 원하는 갯수를 지정
PAGE

news_url <- c()
news_date <-c() 

for (date_i in DATE){
  for (page_i in PAGE){
    dt <- format(as.Date(date_i,origin="1970-01-01"), "%Y.%m.%d")
    naver_url <- paste0(naver_url_1,QUERY,naver_url_2,dt,naver_url_3,dt,naver_url_4,page_i)
    html <- read_html(naver_url)
    temp <- unique(html_nodes(html,'#main_pack')%>% # id= 는 # 을 붙인다
                     html_nodes(css='.group_news')%>%
                     html_nodes(css='.list_news')%>%
                     html_nodes('a')%>%
                     html_attr('href'))
    news_url <- c(news_url,temp)
    news_date <- c(news_date,rep(dt,length(temp)))
  }
  print(dt) # 진행상황을 알기 위함이니 속도가 느려지면 제외
}

NEWS0 <- as.data.frame(cbind(date=news_date, url=news_url, query=QUERY))
NEWS1 <- NEWS0[which(grepl("news.naver.com",NEWS0$url)),]         # 네이버뉴스(news.naver.com)만 대상으로 한다   
NEWS1 <- NEWS1[which(!grepl("sports.news.naver.com",NEWS1$url)),] # 스포츠뉴스(sports.news.naver.com)는 제외한다  
NEWS2 <- NEWS1[!duplicated(NEWS1), ] #


NEWS2$news_title   <- ""
NEWS2$news_content <- ""

for (i in 1:dim(NEWS2)[1]){
  html <- read_html(as.character(NEWS2$url[i]))
  temp_news_title   <- repair_encoding(html_text(html_nodes(html,'#articleTitle')),from = 'utf-8')
  temp_news_content <- repair_encoding(html_text(html_nodes(html,'#articleBodyContents')),from = 'utf-8')
  if (length(temp_news_title)>0){
    NEWS2$news_title[i]   <- temp_news_title
    NEWS2$news_content[i] <- temp_news_content
  }
}

NEWS2$news_content <- gsub("// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback()", "", NEWS2$news_content)
NEWS <- NEWS2


save(NEWS, file="D:/NEWS.RData")


library(multilinguer)


library('tidyverse')
library('reshape2')
library('wordcloud2')
library('rJava') 
Sys.getenv("JAVA_HOME")
library('KoNLP')

load("D:/NEWS.RData")
TEXTFILE <- NEWS$news_content

tbl_TEXTFILE <- TEXTFILE %>% 
  SimplePos09 %>% # 품사구분함수. SimplePos09()는 9개 품사로, SimplePos22()는 22개 품사로 구분 
  melt %>%        # 전체 자료를 tibble 형태로 저장 
  as_tibble %>%   # 전체 자료를 tibble 형태로 저장 
  select(3, 1)    # 실제 분석에 필요한 3열과 1열만 따로 저장 

## 명사형 자료만 골라내어 카운트
tbl_TEXTFILECOUNT0 <- tbl_TEXTFILE %>% # tbl_TEXTFILE 데이터 사용 
  mutate(noun=str_match(value, '([가-힣]+)/N')[,2]) %>% # 명사형(/N) 자료만 골라낸 후 /N 빼고 저장 
  na.omit %>% # 비어있는 값(NA) 제외 
  filter(str_length(noun)>=2) %>%  # '것', '수' 와 같이 별 의미가 없는 의존명사를 제외하기 위하여 한글자 단어는 제외
  count(noun, sort=TRUE)

head(tbl_TEXTFILECOUNT0, 10)


tbl_TEXTFILECOUNT0 <- tbl_TEXTFILE %>% # tbl_TEXTFILE 데이터 사용 
  mutate(noun=str_match(value, '([가-힣]+)/N')[,2]) %>% # 명사형(/N) 자료만 골라낸 후 /N 빼고 저장 
  na.omit %>% # 비어있는 값(NA) 제외 
  filter(str_length(noun)>=2) %>%  # '것', '수' 와 같이 별 의미가 없는 의존명사를 제외하기 위하여 한글자 단어는 제외
  count(noun, sort=TRUE)

tbl_TEXTFILECOUNT1 <- tbl_TEXTFILECOUNT0 %>% filter(n>=5) %>% filter(!noun %in% c(
  "억원", "구청장", "이번", "일대", "주변", 
))



## 200대 키워드만 선정
tbl_TEXTFILECOUNT2 <- tbl_TEXTFILECOUNT1[1:200,] 

# 워드클라우드 그리기
wordcloud2(tbl_TEXTFILECOUNT2,fontFamily="Malgun Gothic", size = 0.5, minRotation=0, maxRotation=0)






