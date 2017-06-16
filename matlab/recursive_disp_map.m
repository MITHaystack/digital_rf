% ----------------------------------------------------------------------------
% Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
% All rights reserved.
%
% Distributed under the terms of the BSD 3-clause license.
%
% The full license is in the LICENSE file, distributed with this software.
% ----------------------------------------------------------------------------
% a function giving a simple example of walking through a map with multiple
% submaps, as returned by DigitalMetadataReader.read.  As the name suggests,
% it calls itself recursively.
%
% $Id$

function recursive_disp_map(map)
    % recursive_disp_map is an example method that walks though multiple
    % layers of containers.Map objects, and displaying them
    % Input: map - containers.Map whose values are may also be further
    % containers.Map objects that desend any number of levels
    disp('Displaying a container.Map object');
    keys = map.keys();
    disp(sprintf('Displaying map with %i keys', length(keys)));
    for i=1:length(keys)
        key = keys{i};
        disp('This key is:');
        disp(key);
        value = map(key);
        if (isprop(value,'ValueType'))
            disp('Value is another map');
            recursive_disp_map(value);
        else
            disp('Value is:');
            disp(value);
        end
    end
end
