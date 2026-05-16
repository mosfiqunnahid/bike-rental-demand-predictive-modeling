############################################################
# Predictive Modeling of Bike Rental Demand
# Dataset: UCI Bike Sharing Dataset
# Focus: Regression Analysis, Polynomial Modeling,
#        AIC Model Selection, and Diagnostics
############################################################

# 1. Packages -------------------------------------------------------------

packages <- c(
  "tidyverse", "janitor", "broom", "car", "MASS", "ggplot2"
)

install.packages(setdiff(packages, rownames(installed.packages())))

library(tidyverse)
library(janitor)
library(broom)
library(car)
library(MASS)
library(ggplot2)

# 2. Create Output Folders ------------------------------------------------

dir.create("outputs", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

# 3. Import Dataset -------------------------------------------------------

bike_raw <- read_csv("data/day.csv") %>%
  clean_names()

# 4. Data Preparation -----------------------------------------------------

bike_data <- bike_raw %>%
  mutate(
    season = as.factor(season),
    yr = as.factor(yr),
    mnth = as.factor(mnth),
    holiday = as.factor(holiday),
    weekday = as.factor(weekday),
    workingday = as.factor(workingday),
    weathersit = as.factor(weathersit),
    temp_squared = temp^2
  )

write_csv(bike_data, "outputs/clean_bike_data.csv")

# 5. Exploratory Data Analysis -------------------------------------------

descriptive_stats <- bike_data %>%
  summarise(
    observations = n(),
    mean_count = mean(cnt, na.rm = TRUE),
    median_count = median(cnt, na.rm = TRUE),
    sd_count = sd(cnt, na.rm = TRUE),
    mean_temp = mean(temp, na.rm = TRUE),
    mean_humidity = mean(hum, na.rm = TRUE),
    mean_windspeed = mean(windspeed, na.rm = TRUE)
  )

write_csv(descriptive_stats, "outputs/descriptive_statistics.csv")

p_count_distribution <- ggplot(bike_data, aes(x = cnt)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Distribution of Daily Bike Rental Demand",
    x = "Daily rental count",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  "figures/bike_demand_distribution.png",
  p_count_distribution,
  width = 8,
  height = 5
)

p_temp_demand <- ggplot(bike_data, aes(x = temp, y = cnt)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE) +
  labs(
    title = "Non-Linear Relationship Between Temperature and Bike Demand",
    x = "Normalized temperature",
    y = "Daily rental count"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  "figures/temperature_demand_relationship.png",
  p_temp_demand,
  width = 8,
  height = 5
)

# 6. Regression Models ----------------------------------------------------

model_linear <- lm(
  cnt ~ temp + hum + windspeed + season + weathersit + workingday + holiday,
  data = bike_data
)

model_polynomial <- lm(
  cnt ~ temp + temp_squared + hum + windspeed + season + weathersit + workingday + holiday,
  data = bike_data
)

model_stepwise <- stepAIC(model_polynomial, direction = "both", trace = FALSE)

# 7. Model Outputs --------------------------------------------------------

write_csv(tidy(model_linear), "outputs/linear_model_coefficients.csv")
write_csv(tidy(model_polynomial), "outputs/polynomial_model_coefficients.csv")
write_csv(tidy(model_stepwise), "outputs/final_model_coefficients.csv")

model_comparison <- tibble(
  model = c("Linear Model", "Polynomial Model", "Stepwise AIC Model"),
  aic = c(AIC(model_linear), AIC(model_polynomial), AIC(model_stepwise)),
  r_squared = c(
    summary(model_linear)$r.squared,
    summary(model_polynomial)$r.squared,
    summary(model_stepwise)$r.squared
  ),
  adjusted_r_squared = c(
    summary(model_linear)$adj.r.squared,
    summary(model_polynomial)$adj.r.squared,
    summary(model_stepwise)$adj.r.squared
  )
)

write_csv(model_comparison, "outputs/model_comparison.csv")

# 8. Multicollinearity Assessment ----------------------------------------

vif_results <- vif(model_stepwise)
write_csv(
  as.data.frame(vif_results) %>% rownames_to_column("variable"),
  "outputs/vif_results.csv"
)

# 9. Diagnostic Plots -----------------------------------------------------

png("figures/residual_diagnostics.png", width = 1000, height = 800)
par(mfrow = c(2, 2))
plot(model_stepwise)
dev.off()

# 10. Predicted vs Observed ----------------------------------------------

bike_data$predicted_count <- predict(model_stepwise, newdata = bike_data)

p_predicted <- ggplot(bike_data, aes(x = predicted_count, y = cnt)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Predicted vs Observed Bike Rental Demand",
    x = "Predicted rental count",
    y = "Observed rental count"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  "figures/predicted_vs_observed.png",
  p_predicted,
  width = 8,
  height = 5
)

############################################################
# End of Analysis
############################################################
