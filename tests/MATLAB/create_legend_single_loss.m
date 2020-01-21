function legend_ = create_legend_single_loss(Models, Nonlin)
legend_ = {};
for m_idx = 1:length(Models)
    if contains(Models{m_idx}, 'N')
        for n_idx = 1:length(Nonlin)
            legend_{end+1} = sprintf('%s, Nonlin = %s', strrep(Models{m_idx}, '_','\_'), strrep(Nonlin{n_idx}, '_','\_'));
        end
    else
        legend_{end+1} = sprintf('%s', strrep(Models{m_idx}, '_','\_'));
    end
end

end