function legend_ = create_legend_single_loss(Models)
legend_ = cell(length(Models), 1);
for m_idx = 1:length(Models)
        legend_{m_idx} = sprintf('%s', strrep(Models{m_idx}, '_','\_'));
end

end