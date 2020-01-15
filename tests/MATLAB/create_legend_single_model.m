function legend_ = create_legend_single_model(loss_dB)
legend_ = cell(length(loss_dB),1);

for l_idx = 1:length(loss_dB)
    legend_{l_idx} = sprintf('L/MZI = %.2f dB', loss_dB(l_idx));
end

end