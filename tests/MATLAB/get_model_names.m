function models = get_model_names(Models)

models = cell(length(Models),1);

for ii = 1:length(Models)
    switch Models{ii}
        case 'R_P'
            models{ii} = 'Reck';
        case 'R_D_P'
            models{ii} = 'Reck + DMM';
        case 'C_Q_P'
            models{ii} = 'Diamond';
        case 'R_I_P'
            models{ii} = 'Reck + Inverted Reck';
        case 'R_D_I_P'
            models{ii} = 'Reck + DMM + Inverted Reck';
        case 'C_W_P'
            models{ii} = 'Center Diamond';
        case 'I_P'
            models{ii} = 'Inverted Reck';
        case 'E_P'
            models{ii} = 'Clements';
        otherwise
            models{ii} = 'Unknown Mesh';
    end
end

end