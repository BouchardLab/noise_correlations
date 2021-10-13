colors = {
    'observed': 'black',
    'shuffle': 'gray',
    'uniform': '#343D95',
    'rotation': 'red',
    'fa': 'orchid',
}
titles = ['Retina', 'V1', 'PAC']

ax_label_size = 10
tick_label_size = 8
letter_size = ax_label_size
legend_size = tick_label_size
lw = 1
line_alpha = 0.8
fill_alpha = 0.3
title_pad = 10

# Division between supoptimal, chance, optimal
plot_lower = 1. / 3.
plot_upper = 2. / 3.

# Statistics to take after subselections
stats_frac_lower = 40.
stats_frac_upper = 60.
stats_frac_middle = 0.5 * (stats_frac_upper + stats_frac_lower)
ci = .99

# Which fraction to subselect
select_lower = 90
select_upper = 100

# Percentiles to directly subselect
p_lower = select_lower + stats_frac_lower / 10.
p_upper = select_lower + stats_frac_upper / 10.
p_middle = 0.5 * (p_lower + p_upper)