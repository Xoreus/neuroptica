function legend_ = create_legend_trainingLoss(Models, losses_dB_training)
legend_ = {};
for m_idx = 1:length(Models)
    for l_idx = 1:length(losses_dB_training)
        legend_{end+1} = sprintf('%s, trainig loss = %.3f dB', strrep(Models{m_idx}, '_','\_'), losses_dB_training(l_idx));
    end
end

end