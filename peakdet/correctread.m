function [corrected] = correctread (raw)
% To correct the result of dlmread

% creating a variable containing number of columns present in matrix, to
% know how many columns to integrate into results matrix ("data") 
nbcol = 0;

treatedline = 1;
for m = 1:length(raw(:,1))
   LINESHORT = [];
   LINESHORT = nonzeros(raw(m,:));
   if length(LINESHORT) > nbcol
       nbcol = length (LINESHORT);
   end
   if length(LINESHORT) < 4
      ['WARNING: input had to be corrected:']
      ['Extra line in file. Check file.']
   else 
      % condition: if the line has 5 values and the Oq value is 13, 
      % then it is due to a mistake at loading: dlmread yields value 13
      % when extra delimiters are present at end of line (for unknown reasons).
      % An extra condition must be added: length of the line == 5, 
      % otherwise the loop may be entered when the length is 4, 
      % and the programme crashes.
      if length(LINESHORT) == 5
         if LINESHORT(5) == 13
         disp(['WARNING: mistaken Oq value in file ',num2str(i),', line ',num2str(m)])
         disp(['Extra space at end of line corrected at loading. Check file.'])
            for o = 1:4
               dataNZ(treatedline,o) = LINESHORT(o);
            end
            treatedline = treatedline + 1;
         else
            ['DEOPA value missing in file ',num2str(i),', line ',num2str(m)]
            for o = 1:4
               dataNZ(treatedline,o) = LINESHORT(o);
            end
            treatedline = treatedline + 1;
         end
      else
        for o = 1:length(LINESHORT)
           dataNZ(treatedline,o) = LINESHORT(o);
        end
        treatedline = treatedline + 1;
      end
   end
end
corrected = dataNZ;