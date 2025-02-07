install.packages('foreign') 
install.packages('dplyr') 
install.packages('psych')
install.packages('lavaan') 

library(foreign)
library(dplyr)
library(psych)
library(lavaan)

setwd()

dataset <- read.csv("dataset.csv")
head(dataset) 
attach(dataset) 

ISSP_EU <- subset(dataset, COUNTRY != "AU-Australia" & COUNTRY != "CA-Canada" & COUNTRY != "TW-Taiwan" & COUNTRY !="IL-Israel" &
                    COUNTRY !="JP-Japan" & COUNTRY != "KR-Korea (South)" & COUNTRY != "NZ- New Zealand" &
                    COUNTRY !="PH-Philippines" & COUNTRY != "RU-Russia" & COUNTRY != "ZA-South Africa" & COUNTRY != "US-United States")

#CFA

#Факторная модель для измерения антииммигрантских установок
model_cfa <- '
anti =~ v48 + v42 + v43 + v44 + v45 
'
fit <- cfa(model_cfa, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v48", "v42", "v43", "v44", "v45"), group = "YEAR_SDNO")
summary(fit, standardized = T)
fitmeasures(fit, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea"))

fit.1 <- cfa(model_cfa, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v48", "v42", "v43", "v44", "v45"), group = "YEAR_SDNO", 
             group.equal = "loadings")
summary(fit.1, standardized = T)
fitmeasures(fit.1, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea"))

fit.2 <- cfa(model_cfa, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v48", "v42", "v43", "v44", "v45"), group = "YEAR_SDNO",
             group.equal = c("loadings", "intercepts"))
summary(fit.2, standardized = T)
fitmeasures(fit.2, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea"))

#Факторная модель для измерения национальной идентичности
model_cfa1 <- '
cult =~ v18 + v25 + v26 + v28 
polit =~ v20 + v21 + v22 + v23 + v29 
blind =~ v15 + v16 + v32
ethn =~ v5 + v7 + v8 + v9  
'
fit1 <- cfa(model_cfa1, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v18", "v25", "v26", "v28",
                                                                                         "v20", "v21", "v22", "v23", "v29",
                                                                                         "v15", "v16", "v32",
                                                                                         "v5", "v7", "v8", "v9"), group = "YEAR_SDNO")
summary(fit1, standardized = T)
fitmeasures(fit1, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea"))

fit1.1 <- cfa(model_cfa1, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v18", "v25", "v26", "v28",
                                                                                           "v20", "v21", "v22", "v23", "v29",
                                                                                           "v15", "v16", "v32",
                                                                                           "v5", "v7", "v8", "v9"), group = "YEAR_SDNO",
              group.equal = "loadings")
summary(fit1.1, standardized = T)
fitmeasures(fit1.1, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea"))

fit1.2 <- cfa(model_cfa1, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v18", "v25", "v26", "v28",
                                                                                           "v20", "v21", "v22", "v23", "v29",
                                                                                           "v15", "v16", "v32",
                                                                                           "v5", "v7", "v8", "v9"), group = "YEAR_SDNO",
              group.equal = c("loadings", "intercepts"))
summary(fit1.2, standardized = T)
fitmeasures(fit1.2, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea"))

alpha(cov(ISSP_EU[,c("v48","v42","v43","v44","v45")]))
alpha(cov(ISSP_EU[,c("v20","v21","v22","v23","v29")]))
alpha(cov(ISSP_EU[,c("v18","v25","v26", "v28")]))
alpha(cov(ISSP_EU[,c("v5","v7", "v8", "v9")]))
alpha(cov(ISSP_EU[,c("v15", "v16", "v32")]))

#SEM
model <- '
#Измерительная часть
anti =~ v48 + v42 + v43 + v44 + v45 
cult =~ v18 + v25 + v26 + v28 
polit =~ v20 + v21 + v22 + v23 + v29
ethn =~ v5 + v7 + v8 + v9 
blind =~ v15 + v16 + v32
#Структурная часть
anti ~ ethn + b*polit + cult + c*ECO + blind + DEGREE + SEX + AGE
polit ~ a*ECO + DEGREE
blind ~ DEGREE
ethn ~ DEGREE
cult ~ DEGREE
indirect := a*b
total := c + (a*b)
polit ~~ ethn + cult + blind
ethn ~~ cult + blind
cult ~~ blind
'

fit2 <- sem(model, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v48", "v43", "v45", "v42", "v44",
                                                                                    "v18", "v25", "v26", "v28",
                                                                                    "v20", "v21", "v22", "v23", "v29",
                                                                                    "v15", "v16", "v32",
                                                                                    "v5", "v7", "v8", "v9", "ECO", "DEGREE", "SEX"), group = "YEAR_SDNO")
summary(fit2, fit = T, standardized = T, rsquare = T, ci = T)
fitmeasures(fit2, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea")) 

fit2.1 <- sem(model, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v48", "v43", "v45", "v42", "v44",
                                                                                    "v18", "v25", "v26", "v28",
                                                                                    "v20", "v21", "v22", "v23", "v29",
                                                                                    "v15", "v16", "v32",
                                                                                    "v5", "v7", "v8", "v9", "ECO", "DEGREE", "SEX"), 
            group = "YEAR_SDNO", group.equal = "loadings")
summary(fit2.1, fit = T, standardized = T, rsquare = T, ci = T)
fitmeasures(fit2.1, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea")) 

fit2.2 <- sem(model, data = ISSP_EU, mimic = "MPlus", estimator = "DWLS", ordered = c("v48", "v43", "v45", "v42", "v44",
                                                                                      "v18", "v25", "v26", "v28",
                                                                                      "v20", "v21", "v22", "v23", "v29",
                                                                                      "v15", "v16", "v32",
                                                                                      "v5", "v7", "v8", "v9", "ECO", "DEGREE", "SEX"), 
              group = "YEAR_SDNO", group.equal = c("loadings", "intercepts"))
summary(fit2.2, fit = T, standardized = T, rsquare = T, ci = T)
fitmeasures(fit2.2, c("chisq", "gfi", "agfi","cfi", "tli", "srmr", "rmsea")) 


