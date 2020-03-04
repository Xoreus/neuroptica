function legend_ = create_legend_single_loss(Models)
legend_ = cell(size(Models, 1), 1);
for m_idx = 1:size(Models, 1)
        legend_{m_idx} = sprintf('%s', strrep(Models(m_idx, :), ' ',''));
end

end